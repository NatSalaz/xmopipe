import time
import cv2
import os
import numpy as np
from contextlib import nullcontext

"""
Video segmentation utilities for human detection and tracking.
Handles video I/O, metadata writing, and visual overlays.
"""


# Video processing subfunctions returns None if video is invalid (no FPS or too short)
def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count < fps:
        cap.release()
        return None, 0, 0
    return cap, fps, frame_count


# Converts frame numbers to HH:MM:SS.mmm timestamps and writes to metadata file
# Flags: 'crowded' (multiple people) or 'weird position' (unusual pose)
def write_metadata(
    metadata_path,
    out_path,
    segment_start_frame,
    written_frame_count,
    fps,
    scene_offset,
    flag_crowded,
    flag_weird,
    lock,
):
    start_seconds = segment_start_frame / fps + scene_offset
    end_seconds = (segment_start_frame + written_frame_count) / fps + scene_offset
    start_timestamp = (
        time.strftime("%H:%M:%S.", time.gmtime(start_seconds))
        + f"{int((start_seconds % 1)*1000):03d}"
    )
    end_timestamp = (
        time.strftime("%H:%M:%S.", time.gmtime(end_seconds))
        + f"{int((end_seconds % 1)*1000):03d}"
    )

    flag_text = ""
    if flag_crowded or flag_weird:
        flags = []
        if flag_crowded:
            flags.append("crowded")
        if flag_weird:
            flags.append("weird position")
        flag_text = f" flag: {', '.join(flags)}"

    if lock:
        with lock:
            with open(metadata_path, "a") as f:
                f.write(
                    f"\nExtrait {os.path.basename(out_path)} : {start_timestamp} - {end_timestamp}{flag_text}"
                )


def start_new_segment(video_out_dir, video_basename, seg_idx, fps, frame):
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(video_out_dir, f"{video_basename}_{seg_idx:03d}.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    return out, out_path


# Deletes segment if shorter than MIN_SEGMENT_FRAME_COUNT, otherwise saves metadata and closes the segment with the reason given by trigger_reason
def close_segment(
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
    flag_weird,
    lock,
    frame_buffer,
    trigger_reasons=None,
):
    out_video.release()
    cap_out = cv2.VideoCapture(out_path)
    written_frame_count = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_out.release()

    if verbose and trigger_reasons:
        print(
            f"[{video_basename}] Segment {seg_idx:03d} closed due to: {', '.join(trigger_reasons)}"
        )

    if written_frame_count < MIN_SEGMENT_FRAME_COUNT:
        if os.path.exists(out_path):
            os.remove(out_path)
        if verbose:
            print(
                f"[{video_basename}] Segment removed (only {written_frame_count} frames, required {MIN_SEGMENT_FRAME_COUNT})."
            )
    else:
        write_metadata(
            metadata_path,
            out_path,
            segment_start_frame,
            written_frame_count,
            fps,
            scene_offset,
            flag_crowded,
            flag_weird,
            lock,
        )

    frame_buffer.clear()


# Alternative close function with explicit frame count tracking
def safe_close_segment(
    out_video,
    out_path,
    segment_start_frame,
    fps,
    scene_offset,
    metadata_path,
    video_basename,
    seg_idx,
    verbose,
    min_segment_length,
    flag_crowded,
    flag_weird,
    lock=None,
    trigger_reasons=None,
    frame_written_count=None,
):

    if frame_written_count is not None and frame_written_count < min_segment_length:
        out_video.release()
        if os.path.exists(out_path):
            os.remove(out_path)
        return

    out_video.release()

    # Compute timecodes
    start_seconds = segment_start_frame / fps + scene_offset
    end_seconds = (segment_start_frame + frame_written_count) / fps + scene_offset
    start_timestamp = (
        time.strftime("%H:%M:%S.", time.gmtime(start_seconds))
        + f"{int((start_seconds % 1)*1000):03d}"
    )
    end_timestamp = (
        time.strftime("%H:%M:%S.", time.gmtime(end_seconds))
        + f"{int((end_seconds % 1)*1000):03d}"
    )

    # Compose flags
    flag_text = ""
    if flag_crowded or flag_weird:
        flags = []
        if flag_crowded:
            flags.append("crowded")
        if flag_weird:
            flags.append("weird position")
        flag_text = f" flag: {', '.join(flags)}"

    line = f"\nExtrait {os.path.basename(out_path)} : {start_timestamp} - {end_timestamp}{flag_text}"
    with lock if lock else nullcontext():
        with open(metadata_path, "a") as f:
            f.write(line)

    if verbose:
        print(
            f"[{video_basename}] Segment {seg_idx:03d} written → {frame_written_count} frames."
        )
        if trigger_reasons:
            print(f" ↳ Reason: {', '.join(trigger_reasons)}")


# COCO keypoint skeleton connections to draw on a frame
def draw_skeleton(frame, bboxes, current_ids, kps_all, skeleton, confs):
    if not skeleton:
        return
    skeleton_connections = [
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    ]
    for idx, box in enumerate(bboxes):
        x, y, w, h = box
        cv2.rectangle(
            frame,
            (int(x - w / 2), int(y - h / 2)),
            (int(x + w / 2), int(y + h / 2)),
            (0, 255, 0),
            2,
        )
        text_position = (int(x - w / 2 + 10), int((y - h / 2) + 70))
        area = w * h
        cv2.putText(
            frame,
            (
                "ID: "
                + str(current_ids[idx])
                + " Conf:"
                + str(round(confs[idx], 3))
                + " Size:"
                + str(area)
            ),
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            3,
        )

    for keypoints in kps_all:
        for connection in skeleton_connections:
            if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                pt1 = keypoints[connection[0]]
                pt2 = keypoints[connection[1]]
                if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
                    continue
                cv2.line(
                    frame,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    (0, 255, 0),
                    3,
                )
    for keypoints in kps_all:
        for idx, kp in enumerate(keypoints):
            if kp is None:
                continue
            if idx == 1 or idx == 2:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 8, (0, 255, 0), 3)
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, (0, 255, 0), 3)
            else:
                cv2.circle(
                    frame,
                    (int(kp[0]), int(kp[1])),
                    5,
                    (
                        0,
                        255,
                        0,
                    ),
                    5,
                )


# Ends segment if target missing OR no humans detected for x seconds
def should_end_segment(non_human_frame_count, target_missing_count, fps, recording):
    return (non_human_frame_count >= fps or target_missing_count >= fps) and recording


# Intersection over Union for bounding boxes (format: x, y, w, h)
def compute_iou(box1, box2):
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(inter_x_max - inter_x_min, 0) * max(inter_y_max - inter_y_min, 0)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# Average Euclidean distance between corresponding keypoints
def compute_keypoints_displacement(kp1, kp2):
    return np.mean(np.linalg.norm(np.array(kp1) - np.array(kp2), axis=1))


# Farneback optical flow normalized by ROI diagonal (measures motion intensity)
def compute_normalized_optical_flow_roi(prev_frame, curr_frame, bbox):
    # bbox (x, y, w, h)
    x, y, w_bbox, h_bbox = bbox
    h_img, w_img = prev_frame.shape[:2]
    x_end = int(min(x + w_bbox, w_img))
    y_end = int(min(y + h_bbox, h_img))
    x = int(max(x, 0))
    y = int(max(y, 0))

    if x >= x_end or y >= y_end:
        return 0.0

    prev_roi = cv2.cvtColor(prev_frame[y:y_end, x:x_end], cv2.COLOR_BGR2GRAY)
    curr_roi = cv2.cvtColor(curr_frame[y:y_end, x:x_end], cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_roi,
        curr_roi,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_flow = np.mean(mag)
    diagonal_roi = np.sqrt((x_end - x) ** 2 + (y_end - y) ** 2)
    normalized_flow = mean_flow / diagonal_roi if diagonal_roi > 0 else 0.0
    return normalized_flow


# Hue-Saturation histogram for scene change detection
def calc_frame_hist(frame, bins=(50, 60)):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist
