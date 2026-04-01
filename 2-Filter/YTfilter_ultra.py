import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import os
import re
import cv2
import numpy as np
import threading
import shutil
import argparse
import gc
import io
import time
import traceback
import datetime
import torch
import yaml
import contextlib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from itertools import repeat
from pathlib import Path
from multiprocessing import Manager
from video_verif import VideoTracker, Log
from collections import deque
from ultralytics import YOLO
import cProfile, pstats
import sys

sys.stdout.reconfigure(line_buffering=True)

from processvidutils import (
    initialize_video_capture,
    write_metadata,
    start_new_segment,
    close_segment,
    safe_close_segment,
    draw_skeleton,
    compute_iou,
    compute_normalized_optical_flow_roi,
    calc_frame_hist,
    compute_keypoints_displacement,
)

_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yml"))["filter"]

# Files and params

OPENPOSE_SKELETON = False
MODE = "lightweight"
BACKEND = "onnxruntime"

# Scene segmentation thresholds (from config.yml)
OPTICAL_FLOW_THR = _cfg["optical_flow_threshold"]
LOW_FLOW_THR = _cfg["low_flow_threshold"]


# Global device variable
def init_tracker(gpu_id):
    # CUDA Imports here in order to get the right gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print("Is cuda available: ", torch.cuda.is_available())
    torch.cuda.set_device(gpu_id)  #  visible GPU will be 0 in this process
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Process launched on GPU {gpu_id}. Uses: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )

    # Initialisation tracker
    get_pose_tracker()


# thread-local loading pose tracker
thread_local = threading.local()


def get_pose_tracker():
    with contextlib.redirect_stdout(io.StringIO()):
        thread_local.model = YOLO(_cfg["pose_model"])
    return thread_local.model


def safe_model_track(model, frame, device, **kwargs):
    """Wrapper for YOLO model.track that skips problematic frames."""
    try:
        results = model.track(frame, device=device, **kwargs)
        return results
    except Exception as e:
        msg = str(e).lower()
        if "not enough matching points" in msg:
            Log.warning("[SKIP FRAME] YOLO warning: not enough matching points")
            return None
        else:
            Log.error(f"[TRACK ERROR] {e}")
            raise


def process_video(
    video_path,
    verbose=False,
    skeleton=False,
    lock=None,
    input_dir="cut_videos",
    output_dir="filteredVideos",
    min_bbox_area=50000,
    max_segment_length=500,
):
    cap, fps, frame_count = initialize_video_capture(video_path)
    if cap is None:
        return
    BODY_THR = _cfg["body_confidence_threshold"]
    MIN_SEGMENT_FRAME_COUNT = int(fps * _cfg["min_segment_duration_sec"])
    NON_HUMAN_FRAME_THRESHOLD = int(fps * _cfg["non_human_tolerance_sec"])
    BAD_BBOX_COUNT_THRESHOLD = int(fps * _cfg["bad_bbox_tolerance_sec"])
    STATIC_FLOW_COUNT_THRESHOLD = int(fps * _cfg["static_flow_tolerance_sec"])
    HIGHFLOW_FRAME_COUNT_THRESHOLD = int(fps * _cfg["high_flow_tolerance_sec"])

    optical_flow_frame_skip = _cfg["optical_flow_frame_skip"]
    OPTICAL_FLOW_THR = _cfg["optical_flow_threshold"]
    LOW_FLOW_THR = _cfg["low_flow_threshold"]

    scene_offset = get_scene_offset(video_path)
    metadata_path = os.path.join(Path(video_path).parent, "metadata.txt")

    rel_folder = os.path.dirname(os.path.relpath(video_path, input_dir))
    video_out_dir = os.path.join(output_dir, rel_folder)
    os.makedirs(video_out_dir, exist_ok=True)

    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # Tracking state
    seg_idx = 0
    out_video = None
    out_path = None
    segment_start_frame = 0
    recording = False
    previous_boxes = {}
    previous_keypoints = {}
    updated_ids_map = {}
    new_id_counter = 10000
    score_history = {}
    final_target_ids = None
    target_missing_count = 0
    human_frame_count = 0
    non_human_frame_count = 0
    prev_frame = None
    flow_frame_counter = 0
    flag_crowded = False
    flag_pose_anomaly = False
    frame_batch = deque()
    frozen_count = 0
    previous_detection = None
    frames_to_delete = 0
    bad_bbox_count = 0
    frame_written_count = 0
    CUT_MARGIN = int(fps * _cfg["cut_margin_sec"])
    # infer_time = 0
    # end_time = 0

    try:
        model = get_pose_tracker()
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    while cap.isOpened():
        try:
            success, frame = cap.read()
            if not success:
                break
        except Exception as e:
            Log.error(f"Frame read failed: {e}")
            break

        shot_boundary_triggered = False
        trigger_reasons = []

        try:
            results = safe_model_track(
                model,
                frame,
                device=DEVICE,
                persist=True,
                verbose=False,
                conf=BODY_THR,
                iou=_cfg["yolo_tracking_iou"],
            )
            if results is None:
                if results is None:
                    trigger_reasons.append("Skipping due to YOLO matching points issue")
                    if recording:
                        frames_to_delete = min(len(frame_batch), 15)
                        for _ in range(frames_to_delete):
                            if frame_batch:
                                frame_batch.pop()
                        while frame_batch:
                            out_video.write(frame_batch.popleft())
                            frame_written_count += 1
                        safe_close_segment(
                            out_video,
                            out_path,
                            segment_start_frame,
                            fps,
                            scene_offset,
                            metadata_path,
                            video_basename,
                            seg_idx,
                            verbose,
                            MIN_SEGMENT_FRAME_COUNT,
                            flag_crowded,
                            flag_pose_anomaly,
                            lock,
                            trigger_reasons=trigger_reasons,
                            frame_written_count=frame_written_count,
                        )
                        recording = False
                        frame_written_count = 0
                        frame_batch.clear()
                    continue
            result = results[0] if isinstance(results, list) else results
        except Exception as e:
            Log.warning(f"[Inference skipped] {e}")
            continue

        human = (
            result.boxes
            and result.boxes.id is not None
            and (result.keypoints.conf.mean(axis=1).cpu().numpy() > 0.5).any().item()
        )
        if human:
            non_human_frame_count = 0
            human_frame_count += 1
            confs = result.boxes.conf.cpu().numpy()
            bboxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            if result.keypoints.conf is not None:
                meanscores = result.keypoints.conf.mean(axis=1).cpu().numpy()
            else:
                meanscores = np.array([0.0] * len(track_ids))
            kps_all = (
                result.keypoints.xy.cpu().numpy()
                if result.keypoints.xy is not None
                else [None] * len(track_ids)
            )
            current_ids = track_ids.copy()

            max_area = 0

            # Re-ID logic
            for i in range(len(track_ids)):
                original_id = int(track_ids[i])
                mapped_id = updated_ids_map.get(original_id, original_id)
                current_box = bboxes[i]

                current_area = current_box[2] * current_box[3]
                max_area = max(max_area, current_area)

                current_kps = kps_all[i]
                change_id = False

                if mapped_id in previous_boxes:
                    prev_box = previous_boxes[mapped_id]
                    iou = compute_iou(current_box, prev_box)
                    area_ratio = (
                        current_area / (prev_box[2] * prev_box[3])
                        if prev_box[2] * prev_box[3] != 0
                        else 0
                    )
                    if iou < _cfg["reid_iou_threshold"] or not (_cfg["reid_area_ratio_min"] <= area_ratio <= _cfg["reid_area_ratio_max"]):
                        change_id = True

                if current_kps is not None and mapped_id in previous_keypoints:
                    kp_disp = compute_keypoints_displacement(
                        current_kps, previous_keypoints[mapped_id]
                    )
                    if kp_disp > _cfg["reid_keypoint_displacement"]:
                        change_id = True

                if change_id:
                    new_id_counter += 1
                    mapped_id = new_id_counter
                    updated_ids_map[original_id] = mapped_id

                current_ids[i] = mapped_id
                previous_boxes[mapped_id] = current_box
                if current_kps is not None:
                    previous_keypoints[mapped_id] = current_kps
            # Freeze detection logic
            current_detection = {
                "bboxes": np.round(bboxes, 2).tolist(),
                "keypoints": (
                    np.round(kps_all, 2).tolist()
                    if isinstance(kps_all, np.ndarray)
                    else None
                ),
            }

            if (
                previous_detection is not None
                and current_detection == previous_detection
            ):
                frozen_count += 1
            else:
                frozen_count = 0

            previous_detection = current_detection

            # Cut trigger if frozen for too long
            if frozen_count >= _cfg["frozen_frames_threshold"]:
                shot_boundary_triggered = True
                trigger_reasons.append("Frozen detection (same bbox/keypoints 10x)")
                frozen_count = 0

            # Score tracking
            if human_frame_count <= fps:
                for i, tid in enumerate(current_ids):
                    score_history.setdefault(tid, []).append(meanscores[i])
                if human_frame_count == fps:
                    avg_scores = {
                        tid: np.mean(vals) for tid, vals in score_history.items()
                    }
                    final_target_ids = sorted(
                        [tid for tid, avg in avg_scores.items() if avg > BODY_THR],
                        key=avg_scores.get,
                        reverse=True,
                    )
                    if verbose:
                        print(
                            f"[{video_basename}: Seg {seg_idx:03d}] Important target IDs:",
                            final_target_ids,
                        )
                    if final_target_ids == []:
                        human_frame_count = 0
                        non_human_frame_count = NON_HUMAN_FRAME_THRESHOLD
                        final_target_ids = None

            # Crowd + pose_anomaly detection
            if result.keypoints and result.keypoints.xy is not None:
                num_persons = result.keypoints.xy.shape[0]
                if num_persons > _cfg["crowd_person_threshold"]:
                    flag_crowded = True
                for i in range(num_persons):
                    kps = result.keypoints.xy[i]
                    scs = (
                        result.keypoints.conf[i]
                        if result.keypoints.conf is not None
                        else None
                    )
                    if scs is not None and scs[0] > BODY_THR and scs[15] > BODY_THR:
                        if kps[0][1] > kps[15][1]:
                            flag_pose_anomaly = True
                    if scs is not None and scs[0] > BODY_THR and scs[16] > BODY_THR:
                        if kps[0][1] > kps[16][1]:
                            flag_pose_anomaly = True
            if max_area < min_bbox_area:
                bad_bbox_count += 1
                if bad_bbox_count >= BAD_BBOX_COUNT_THRESHOLD:
                    frames_to_delete = min(bad_bbox_count, len(frame_batch))
                    bad_bbox_count = 0
                    shot_boundary_triggered = True
                    trigger_reasons.append(f"BBox too small: max_area={max_area}")
            else:
                bad_bbox_count = 0

            # Optical flow check
            if (
                prev_frame is not None
                and flow_frame_counter % optical_flow_frame_skip == 0
            ):
                bbox = np.mean(bboxes, axis=0) if len(bboxes) > 0 else None
                if bbox is not None:
                    flow_norm = compute_normalized_optical_flow_roi(
                        prev_frame, frame, bbox
                    )
                    if flow_norm > OPTICAL_FLOW_THR:
                        high_flow_count += 1
                        static_flow_count = 0
                        if high_flow_count >= HIGHFLOW_FRAME_COUNT_THRESHOLD:
                            frames_to_delete = min(high_flow_count, len(frame_batch))
                            shot_boundary_triggered = True
                            trigger_reasons.append(f"High flow: {flow_norm}")
                            high_flow_count = 0
                    elif flow_norm < LOW_FLOW_THR:
                        static_flow_count += 1
                        high_flow_count = 0
                        if static_flow_count >= STATIC_FLOW_COUNT_THRESHOLD:
                            frames_to_delete = min(static_flow_count, len(frame_batch))
                            shot_boundary_triggered = True
                            trigger_reasons.append("Static segment")
                            static_flow_count = 0

            prev_frame = frame.copy()
            if not recording:
                frame_written_count = 0
                seg_idx += 1
                out_video, out_path = start_new_segment(
                    video_out_dir, video_basename, seg_idx, fps, frame
                )
                recording = True
                segment_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                static_flow_count = 0
                high_flow_count = 0
                human_frame_count = 0
                final_target_ids = None
                score_history = {}
        else:
            target_missing_count += 1
            non_human_frame_count += 1
            bad_bbox_count += 1

        # Handle target missing
        if final_target_ids:
            found_valid_target = any(
                tid in final_target_ids and score > BODY_THR
                for tid, score in zip(current_ids, meanscores)
            )
            if not found_valid_target:
                target_missing_count += 1
                if target_missing_count >= fps:
                    trigger_reasons.append("No tracked target visible")

                    if recording:
                        # Remove last target_missing_count frames from batch
                        for _ in range(min(target_missing_count, len(frame_batch))):
                            if frame_batch:
                                frame_batch.pop()

                        # Write the other frames
                        while frame_batch:
                            out_video.write(frame_batch.popleft())
                            frame_written_count += 1

                        safe_close_segment(
                            out_video,
                            out_path,
                            segment_start_frame,
                            fps,
                            scene_offset,
                            metadata_path,
                            video_basename,
                            seg_idx,
                            verbose,
                            MIN_SEGMENT_FRAME_COUNT,
                            flag_crowded,
                            flag_pose_anomaly,
                            lock,
                            trigger_reasons=trigger_reasons,
                            frame_written_count=frame_written_count,
                        )
                        recording = False
                        frame_written_count = 0
                        frame_batch.clear()
                        continue

        if frame_written_count >= max_segment_length:
            shot_boundary_triggered = True
            trigger_reasons.append(f"Segment exceeded {max_segment_length} frames")

        # Record frame
        if recording:
            draw_skeleton(frame, bboxes, current_ids, kps_all, skeleton, confs)
            # We work via frame_batch, we will write as we go
            # if we exceed the interval, we flush and write to the video
            frame_batch.append(frame)
            if not shot_boundary_triggered and len(frame_batch) > CUT_MARGIN:
                # We keep the last "guilty" frames in buffer
                while len(frame_batch) > CUT_MARGIN:
                    out_video.write(frame_batch.popleft())
                    frame_written_count += 1

        # Segment termination
        if (
            shot_boundary_triggered
            or non_human_frame_count >= NON_HUMAN_FRAME_THRESHOLD
        ) and recording:
            if non_human_frame_count >= NON_HUMAN_FRAME_THRESHOLD:
                trigger_reasons.append("No human body since too much frames")
                frames_to_delete = min(non_human_frame_count, len(frame_batch))
            for _ in range(frames_to_delete):
                if frame_batch:
                    frame_batch.pop()
            while frame_batch:
                out_video.write(frame_batch.popleft())
                frame_written_count += 1
            safe_close_segment(
                out_video,
                out_path,
                segment_start_frame,
                fps,
                scene_offset,
                metadata_path,
                video_basename,
                seg_idx,
                verbose,
                MIN_SEGMENT_FRAME_COUNT,
                flag_crowded,
                flag_pose_anomaly,
                lock,
                trigger_reasons=trigger_reasons,
                frame_written_count=frame_written_count,
            )

            recording = False
            frame_written_count = 0

            recording = False
            static_flow_count = 0
            high_flow_count = 0
            human_frame_count = 0
            final_target_ids = None
            score_history = {}
            flag_crowded = False
            flag_pose_anomaly = False
        flow_frame_counter += 1

    # Final flush after video ends
    if recording and out_path and os.path.exists(out_path):
        frames_to_delete = target_missing_count
        for _ in range(min(frames_to_delete, len(frame_batch))):
            if frame_batch:
                frame_batch.pop()
        while frame_batch:
            out_video.write(frame_batch.popleft())
            frame_written_count += 1
        safe_close_segment(
            out_video,
            out_path,
            segment_start_frame,
            fps,
            scene_offset,
            metadata_path,
            video_basename,
            seg_idx,
            verbose,
            MIN_SEGMENT_FRAME_COUNT,
            flag_crowded,
            flag_pose_anomaly,
            lock,
            trigger_reasons=["End of video"],
            frame_written_count=frame_written_count,
        )
    cap.release()
    if hasattr(thread_local, "model"):
        del thread_local.model
    torch.cuda.empty_cache()
    gc.collect()


# Multiprocessing with tracking
def process_video_with_tracking(args):
    (
        video_path,
        verbose,
        skeleton,
        lock,
        keep,
        input_dir,
        output_dir,
        min_bbox_area,
        max_segment_length,
    ) = args

    local_tracker = VideoTracker(input_dir)
    local_tracker.start_heartbeat()
    try:
        video_path_obj = Path(video_path).resolve()
        if not local_tracker.is_video_available(
            video_path
        ) or local_tracker.is_video_failed(video_path):
            if verbose:
                print(
                    f"[Tracker] Video {video_path} already processed, in progress or failed."
                )
            return
        if not local_tracker.mark_video_in_progress(video_path):
            print(f"[Tracker] Impossible to reserve {video_path} for treatment.")
            return

        if verbose:
            print(f"[Tracker] Vidéo marked as in progress : {video_path}")
        start_time = time.time()
        process_video(
            video_path,
            verbose,
            skeleton,
            lock,
            input_dir,
            output_dir,
            min_bbox_area,
            max_segment_length,
        )
        processing_time = time.time() - start_time
        metadata = {
            "processing_time": processing_time,
            "date_processed": datetime.datetime.now().isoformat(),
            "tracks": None,  # we could eventually add infos about track here
        }
        parent_folder = video_path_obj.parent
        copy_metadata(str(parent_folder), output_dir)
        folder_completed = local_tracker.mark_video_processed(video_path, metadata)
        if verbose:
            print(
                f"[Tracker] Video {video_path} was processed with success in {processing_time:.2f} secondes."
            )
        subfolder_path = str(parent_folder)
        is_fully_processed = local_tracker.is_subfolder_fully_processed(subfolder_path)

        if verbose:
            print(
                f"[DEBUG] folder_completed={folder_completed}, is_fully_processed={is_fully_processed}"
            )
        if folder_completed or is_fully_processed:
            try:
                relative_path = parent_folder.relative_to(Path(input_dir))
            except ValueError:
                relative_path = parent_folder.name
            output_path = Path(output_dir) / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            filter_metadata(str(output_path))
            if verbose:
                print(f"[POSTPROCESS] Folder {parent_folder} completely treated.")
                print(f"[POSTPROCESS] Output path: {output_path}")
    except Exception as e:
        if "matching points" in str(e).lower():
            Log.warning(f"[YOLO WARNING IGNORED] {e}")
        else:
            error_message = f"{e}\n{traceback.format_exc()}"
            local_tracker.mark_video_failed(video_path, error_message)
            print(f"[Tracker] Error while treating {video_path} : {error_message}")
            # remove only the segments generated by this video
            try:
                parent_folder = Path(video_path).resolve().parent
                relative_path = parent_folder.relative_to(Path(input_dir))
            except ValueError:
                relative_path = parent_folder.name

            video_basename = Path(video_path).stem  # ex: video_001
            output_path = Path(output_dir) / relative_path

            if output_path.exists():
                deleted = False
                for file in output_path.glob(f"{video_basename}_*.mp4"):
                    try:
                        file.unlink()
                        print(f"[CLEANUP] removed segment : {file}")
                        deleted = True
                    except Exception as del_err:
                        print(f"[CLEANUP] Error while removing {file} : {del_err}")
                if deleted:
                    print(f"[CLEANUP] Corrupted segments removed in {output_path}")

            # update metadatas
            try:
                filter_metadata(str(output_path))
            except Exception as meta_error:
                print(f"[CLEANUP] Fail when filtering metadatas : {meta_error}")
    finally:
        local_tracker.stop_heartbeat()


def process_pool(
    gpu_id,
    video_list,
    verbose,
    skeleton,
    lock,
    keep,
    input_dir,
    output_dir,
    min_bbox_area=None,
    max_segment_length=None,
):
    if min_bbox_area is None:
        min_bbox_area = _cfg["min_bbox_area"]
    if max_segment_length is None:
        max_segment_length = _cfg["max_segment_length"]
    with ProcessPoolExecutor(
        max_workers=_cfg["workers_per_gpu"], initializer=init_tracker, initargs=(gpu_id,)
    ) as executor:
        args = zip(
            video_list,
            repeat(verbose),
            repeat(skeleton),
            repeat(lock),
            repeat(keep),
            repeat(input_dir),
            repeat(output_dir),
            repeat(min_bbox_area),
            repeat(max_segment_length),
        )
        list(
            tqdm(
                executor.map(process_video_with_tracking, args),
                total=len(video_list),
                desc=f"Processing on GPU {gpu_id}",
            )
        )


def process_videos(
    input_dir,
    output_dir,
    verbose=False,
    skeleton=False,
    lock=None,
    keep=False,
    min_bbox_area=None,
    max_segment_length=None,
):
    if min_bbox_area is None:
        min_bbox_area = _cfg["min_bbox_area"]
    if max_segment_length is None:
        max_segment_length = _cfg["max_segment_length"]
    video_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(input_dir)
        for f in fs
        if f.endswith((".mp4", ".avi", ".mov"))
    ]
    print(f"Treating {len(video_files)} videos.")
    if not video_files:
        print("No video found. End of treatment")
        return
    process_pool(
        0,
        video_files,
        verbose,
        skeleton,
        lock,
        keep,
        input_dir,
        output_dir,
        min_bbox_area,
        max_segment_length,
    )
    print("FINI")


# Metadata functions


def get_scene_offset(video_path):
    """Returns the offset of the scene from metadata.txt"""
    video_name = os.path.basename(video_path)
    video_prefix = video_name.split("-")[0]
    try:
        scene_number = int(video_name.split("-")[1].split(".")[0])
    except (IndexError, ValueError):
        return 0.0
    metadata_path = os.path.join(os.path.dirname(video_path), "metadata.txt")
    if not os.path.exists(metadata_path):
        return 0.0
    with open(metadata_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        match = re.search(
            rf"Scene {scene_number}:\s+(\d+:\d+:\d+\.\d+)\s+-\s+(\d+:\d+:\d+\.\d+)",
            line,
        )
        if match:
            h, m, s = match.group(1).split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def copy_metadata(video_path, output_dir):
    for root, _, files in os.walk(video_path):
        if "metadata.txt" in files:
            rel_folder = os.path.relpath(root, video_path)
            dest_folder = os.path.join(
                output_dir, rel_folder, os.path.basename(video_path)
            )
            os.makedirs(dest_folder, exist_ok=True)
            src = os.path.join(root, "metadata.txt")
            dst = os.path.join(dest_folder, "metadata.txt")
            shutil.copyfile(src, dst)


def filter_metadata(video_path):
    if not os.path.isdir(video_path):
        return
    meta_path = os.path.join(video_path, "metadata.txt")
    if not os.path.exists(meta_path):
        print(f"No metadata.txt found in {video_path}")
        return
    scenes = set()
    sub_path = os.path.join(video_path)
    if not os.path.exists(sub_path):
        return
    for fname in os.listdir(sub_path):
        if fname.endswith((".mp4", ".avi", ".mov")):
            parts = fname.split("_")
            if len(parts) < 2:
                return
            try:
                scene_num = int(parts[1].split("-")[1].split(".")[0])
                scenes.add(scene_num)
            except ValueError:
                return
    if not scenes:
        print(f"No scenes found in {video_path}, deleting folder.")
        shutil.rmtree(video_path)
        return
    with open(meta_path, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if not line.startswith("Scene") and not line.startswith("video_"):
            new_lines.append(line)
    with open(meta_path, "w") as f:
        f.writelines(new_lines)
    print(f"Updated metadata in {meta_path} with scenes: {sorted(scenes)}")
    reorder_and_rename_clips(video_path)
    print(f"Scenes filtered and renamed in {video_path}")


def reorder_and_rename_clips(folder_path):
    metadata_path = os.path.join(folder_path, "metadata.txt")
    if not os.path.exists(metadata_path):
        print(f"No metadata.txt in {folder_path}")
        return

    with open(metadata_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    header = []
    clips = []

    for line in lines:
        if line.strip().startswith("Extrait"):
            match = re.match(
                r"Extrait (.+?) : (\d+:\d+:\d+\.\d+) - (\d+:\d+:\d+\.\d+)(.*)",
                line.strip(),
            )
            if match:
                filename, start, end, extra = match.groups()
                h, m, s = start.split(":")
                seconds = int(h) * 3600 + int(m) * 60 + float(s)
                clips.append((seconds, filename, start, end, extra.strip()))
        else:
            header.append(line)
    clips.sort()

    print(clips)
    for i, (_, old_name, start, end, extra) in enumerate(clips, start=1):
        new_name = f"{i}.mp4"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
        else:
            print(f"Missing file: {old_name}")

        line = f"{new_name} : {start} - {end}"
        if extra:
            line += f", {extra}"
        new_lines.append(line + "\n")
    with open(metadata_path, "w") as f:
        f.writelines(header + ["\n"] + new_lines)
    print(f"Updated metadata.txt for {folder_path}")


# Main


def main():
    parser = argparse.ArgumentParser(description="Process videos with pose estimation.")
    parser.add_argument(
        "--input_root", type=str, default=_cfg["input_dir"], help="Path to input folder."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=_cfg["output_dir"],
        help="Path to output folder.",
    )
    parser.add_argument(
        "--min_bbox_area",
        type=int,
        default=_cfg["min_bbox_area"],
        help="Minimum bounding box area to keep a detection (pixels²).",
    )
    parser.add_argument(
        "--max_segment_length",
        type=int,
        default=_cfg["max_segment_length"],
        help="Maximum number of frames per output segment.",
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep the input folder after processing."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show segmentation reasons."
    )
    parser.add_argument(
        "--skeleton", action="store_true", help="Display detected skeleton in frames."
    )
    args = parser.parse_args()

    with Manager() as manager:
        lock = manager.Lock()
        process_videos(
            input_dir=args.input_root,
            output_dir=args.output_root,
            verbose=args.verbose,
            skeleton=args.skeleton,
            lock=lock,
            keep=args.keep,
            min_bbox_area=args.min_bbox_area,
            max_segment_length=args.max_segment_length,
        )
    print("All treatments done. Script finished properly.")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()
    main()
    total_time = round(time.time() - start_time, 2)
    print("Filtering executed in:", total_time, "s.")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(50)
