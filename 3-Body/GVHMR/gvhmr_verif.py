import argparse
import cProfile, pstats
import copy
import datetime
import gc
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import cv2
import matplotlib.pyplot as plt
import numpy as np
import portalocker
import psutil
import pytorch_lightning as pl
import torch
import functools
import subprocess
import yaml
from omegaconf import OmegaConf
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing.resource_tracker

torch.backends.cudnn.enabled = False
import hydra
from hydra import initialize_config_module, compose
from hmr4d.configs import register_store_gvhmr
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.geo.hmr_cam import (
    convert_K_to_K4,
    estimate_K,
    get_bbx_xys_from_xyxy,
)
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_video_reader,
    get_writer,
    read_video_np,
)
from video_verif import VideoTracker
from merge_bodies import merge_body_npz

_cfg = yaml.safe_load(open(Path(__file__).parent.parent.parent / "config.yml"))["body"]


@functools.lru_cache(maxsize=None)
def get_hmr4d_model(cfg):
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    return model


sys.stdout.reconfigure(line_buffering=True)


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    Log.info(f"Memory use CPU (RSS): {rss_mb:.2f} MB")

    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        Log.info(f"GPU - Allocated memory: {allocated_mb:.2f} MB")
        Log.info(f"GPU - Reserved memory: {reserved_mb:.2f} MB")


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


CRF = _cfg["ffmpeg_crf"]


# ========== Preprocess before gvhmr: yolo,vitpose,vitfeatures and SLAM ==========
@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Starting video: {cfg.video_path}")
    t_total_start = time.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose
    # t_bbox_start = time.time()
    # ---------- 1. Bounding boxes ----------
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        all_tracks, _ = tracker.get_tracks(video_path)
        all_bbx_xyxy = []
        all_bbx_xys = []
        for idx, track_xyxy in enumerate(all_tracks):
            track_xyxy = track_xyxy.float()
            track_xys = get_bbx_xys_from_xyxy(track_xyxy, base_enlarge=_cfg["bbox_enlarge_factor"]).float()
            all_bbx_xyxy.append(track_xyxy)
            all_bbx_xys.append(track_xys)
            if verbose:
                Log.info(f"[Preprocess][Yolo] Track {idx+1}/{len(all_tracks)} processed")
        torch.save({"bbx_xyxy": all_bbx_xyxy, "bbx_xys": all_bbx_xys}, paths.bbx)
        del tracker
    else:
        saved = torch.load(paths.bbx)
        all_bbx_xyxy = saved["bbx_xyxy"]
        all_bbx_xys = saved["bbx_xys"]
        if verbose:
            Log.info(f"[Preprocess][Yolo] Loaded bounding boxes from {paths.bbx}")

    # ---------- 2. VitPose ----------
    valid_indices = []
    all_vitpose = []
    all_flagged_vit = []
    CONF_THRESHOLD = _cfg["vitpose_confidence_threshold"]

    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        for idx, track_xys in enumerate(all_bbx_xys):
            vitpose_track, flagged_vit = vitpose_extractor.extract(video_path, track_xys)
            scores = vitpose_track[..., 2]  # shape: (N_frames, N_joints)
            mean_conf = scores.mean().item()
            if mean_conf >= CONF_THRESHOLD:
                valid_indices.append(idx)
                all_vitpose.append(vitpose_track)
                all_flagged_vit.append(flagged_vit)
                if verbose:
                    Log.info(f"[Preprocess][VitPose] Track {idx+1} accepted (mean conf = {mean_conf:.2f})")
            else:
                if verbose:
                    Log.info(f"[Preprocess][VitPose] Track {idx+1} rejected (mean conf = {mean_conf:.2f})")
        torch.save({"vitpose": all_vitpose, "flagged": all_flagged_vit}, paths.vitpose)
        del vitpose_extractor

        # Also filter bboxes
        all_bbx_xys = [all_bbx_xys[i] for i in valid_indices]
        all_bbx_xyxy = [all_bbx_xyxy[i] for i in valid_indices]
        torch.save({"bbx_xyxy": all_bbx_xyxy, "bbx_xys": all_bbx_xys}, paths.bbx)  # overwrite
    else:
        loadvitpose = torch.load(paths.vitpose)
        all_vitpose = loadvitpose["vitpose"]
        all_flagged_vit = loadvitpose["flagged"]
        if verbose:
            Log.info(f"[Preprocess][VitPose] Loaded 2D poses from {paths.vitpose}")

    # ---------- 3. VitFeatures ----------
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        all_vit_features = []
        for i, track_xys in enumerate(all_bbx_xys):
            vit_features_track = extractor.extract_video_features(video_path, track_xys)
            all_vit_features.append(vit_features_track)
            if verbose:
                Log.info(f"[Preprocess][VitFeatures] Track {i+1} processed")
        torch.save(all_vit_features, paths.vit_features)
        del extractor
    else:
        if verbose:
            Log.info(f"[Preprocess][VitFeatures] Loaded video features from {paths.vit_features}")

    # ---------- 4. SLAM ----------
    if not static_cam:
        t_slam_start = time.time()
        if not Path(paths.slam).exists():
            length, width, height = get_video_lwh(video_path)
            K_fullimg = estimate_K(width, height)
            intrinsics = convert_K_to_K4(K_fullimg)
            slam = SLAMModel(video_path, width, height, intrinsics, buffer=_cfg["slam_buffer"], resize=_cfg["slam_resize"])
            bar = tqdm(total=length, desc="DPVO", disable=True)
            t_loop_start = time.time()
            while True:
                ret = slam.track()
                if ret:
                    bar.update()
                else:
                    break
            t_loop_end = time.time()
            if verbose:
                Log.info(
                    f"[Preprocess][SLAM] Processed {length} frames in {t_loop_end-t_loop_start:.2f}s "
                    f"({(t_loop_end-t_loop_start)/length:.4f} s/frame)"
                )
            slam_results = slam.process()
            Path(paths.slam).parent.mkdir(parents=True, exist_ok=True)
            torch.save(slam_results, paths.slam)
            del slam
        else:
            if verbose:
                Log.info(f"[Preprocess][SLAM] Loaded SLAM results from {paths.slam}")
        Log.info(f"[Preprocess][SLAM] Total SLAM time for {video_path}: {time.time()-t_slam_start:.4f}s")
    else:
        Log.info("[Preprocess][SLAM] Skipped SLAM (static camera)")

    t_total_end = time.time()
    Log.info(f"[Preprocess] Completed in {t_total_end-t_total_start:.2f}s")
    torch.cuda.empty_cache()
    gc.collect()
    return all_vitpose, all_flagged_vit


def load_data_dict(cfg):
    t_start = time.time()
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(paths.slam)
        traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
        R_w2c = quaternion_to_matrix(traj_quat).mT
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    bbx_file = torch.load(paths.bbx)
    vitpose_file = torch.load(paths.vitpose)
    if isinstance(vitpose_file, dict):
        vitpose_file = vitpose_file["vitpose"]
    vitfeat_file = torch.load(paths.vit_features)
    data = {
        "length": torch.tensor(length),
        "all_bbx_xys": bbx_file["bbx_xys"],
        "all_bbx_xyxy": bbx_file["bbx_xyxy"],
        "all_kp2d": vitpose_file,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "all_f_imgseq": vitfeat_file,
    }
    t_end = time.time()
    if cfg.verbose:
        Log.info(f"[load_data_dict] Completed in {t_end-t_start:.4f}s")
    return data


def copy_video_safely(src_video: Path, dest_video: Path, crf=23, max_frames=None):
    lock_file = dest_video.with_suffix(dest_video.suffix + ".lock")
    dest_video.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "w") as lf:
        portalocker.lock(lf, portalocker.LOCK_EX)
        try:
            if dest_video.exists():
                print(f"[copy_video_safely] {dest_video} already exists.")
            else:
                cmd = ["ffmpeg", "-y", "-i", str(src_video)]

                if max_frames:
                    cmd += [
                        "-frames:v",
                        str(max_frames),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-crf",
                        str(crf),
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                    ]
                else:
                    cmd += ["-c", "copy"]

                cmd += [str(dest_video)]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if result.returncode != 0:
                    print(f"[ffmpeg error] {result.stderr}")
                    raise RuntimeError("FFmpeg failed.")
        finally:
            portalocker.unlock(lf)
    if lock_file.exists():
        try:
            lock_file.unlink()
        except Exception as e:
            print(f"Impossible to delete lock file {lock_file}: {e}")


def create_cfg_for_video(video_file: Path, output_root: str, static_cam: bool, verbose: bool, max_frames=None):
    t_start = time.time()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            f"video_name={video_file.stem}",
            f"static_cam={static_cam}",
            f"verbose={verbose}",
        ]
        if output_root is not None:
            overrides.append(f"output_root={output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)
    video_path = Path(video_file)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    if verbose:
        Log.info(f"[create_cfg_for_video] Input: {video_path}")
        Log.info(f"[create_cfg_for_video] (L, W, H) = ({length}, {width}, {height})")
    # builds copy path if diffferent
    cfg.video_path = os.path.join(os.path.dirname(cfg.video_path), video_path.name)
    if verbose:
        Log.info(f"[create_cfg_for_video] Copy Video: {video_file} -> {cfg.video_path}")
    dest_video_path = Path(cfg.video_path)
    # There we copy
    copy_video_safely(video_path, dest_video_path, crf=23, max_frames=max_frames)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    t_end = time.time()
    if verbose:
        Log.info(f"[create_cfg_for_video] Completed for {video_path} in {t_end-t_start:.4f}s")
    return cfg


def cleanup_temp(cfg):
    Log.info("[cleanup_temp] Cleaning temporary directories...")
    path_to_destroy = Path(cfg.output_dir)
    if path_to_destroy.exists():
        try:
            shutil.rmtree(path_to_destroy)
            if cfg.verbose:
                Log.info(f"Removed preprocessing folder {path_to_destroy}")
        except Exception as e:
            tb = traceback.format_exc()
            Log.error(f"Error detroying {path_to_destroy}:\n{tb}")


def preprocess_single_video(cfg):
    Log.info("[preprocess_single_video] Starting pre-processing")
    t_start = time.time()
    all_vitpose, flag_by_tracks = run_preprocess(cfg)
    data = load_data_dict(cfg)
    t_end = time.time()
    Log.info(f"[preprocess_single_video] Completed in {t_end-t_start:.4f}s")
    torch.cuda.empty_cache()
    gc.collect()
    return {"cfg": cfg, "data": data, "all_vitpose": all_vitpose, "flag_by_track": flag_by_tracks}


def inference_single_video(preproc_result, global_hmr4d_model, global_smplx_model):
    Log.info("[inference_single_video] Starting inference")
    t_total_start = time.time()
    cfg = preproc_result["cfg"]
    data = preproc_result["data"]
    all_vitpose = preproc_result["all_vitpose"]
    flag_by_track = preproc_result["flag_by_track"]
    created_output_files = []
    if not Path(cfg.paths.hmr4d_results).exists():
        # Log.info("[HMR4D] Starting prediction") ========== Logs in case you want a lot of verbose
        t_hmr_start = time.time()
        all_pred = []
        for i, (kp2d_track, bbx_track) in enumerate(zip(data["all_kp2d"], data["all_bbx_xys"])):
            t_pred_start = time.time()
            if cfg.verbose:
                Log.info(f"[HMR4D] Predicting for track {i+1}/{len(data['all_kp2d'])}")
            single_data = {
                "length": data["length"],
                "bbx_xys": bbx_track,
                "kp2d": kp2d_track,
                "K_fullimg": data["K_fullimg"],
                "cam_angvel": data["cam_angvel"],
                "f_imgseq": data["all_f_imgseq"][i],
            }
            pred = global_hmr4d_model.predict(single_data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            if isinstance(pred, dict):
                all_pred.append(pred)
            elif isinstance(pred, list):
                all_pred.extend(pred)
            else:
                Log.error(f"Unexpected prediction type: {type(pred)}")
            t_pred_end = time.time()
            if cfg.verbose:
                Log.info(f"[HMR4D] Track {i+1} prediction time: {t_pred_end-t_pred_start:.4f}s")
        torch.save(all_pred, cfg.paths.hmr4d_results)
        t_hmr_end = time.time()
        Log.info(f"[HMR4D] Prediction completed for {cfg.video_path} in {t_hmr_end-t_hmr_start:.2f}s")
    else:
        if cfg.verbose:
            Log.info(f"[HMR4D] Results already exist in {cfg.paths.hmr4d_results}")
    all_preds = torch.load(cfg.paths.hmr4d_results)
    video_name = Path(cfg.video_path).stem
    fps = get_video_fps(cfg.video_path)

    for i, pred in enumerate(all_preds):
        t_smpl_start = time.time()
        smplx_out = global_smplx_model(**pred["smpl_params_global"])
        cam_out = pred["smpl_params_incam"]["transl"]
        transl = pred["smpl_params_global"]["transl"]
        contacts = pred["net_outputs"]["static_conf_logits"]
        global_orient = detach_to_cpu(smplx_out.global_orient).numpy()
        body_pose = detach_to_cpu(smplx_out.body_pose).numpy()
        missing_columns = 9
        zero_padding = np.zeros((body_pose.shape[0], missing_columns))
        left_hand_pose = detach_to_cpu(smplx_out.left_hand_pose).numpy()
        right_hand_pose = detach_to_cpu(smplx_out.right_hand_pose).numpy()
        full_body_pose = np.concatenate(
            [global_orient, body_pose, zero_padding, left_hand_pose, right_hand_pose], axis=1
        )
        bbox_xyxy_np = detach_to_cpu(data["all_bbx_xyxy"][i]).numpy()
        smplx_params_np = {
            "vitpose": all_vitpose[i],
            "flagged_frames": flag_by_track[i],
            "trans": transl,
            "cam_transl": detach_to_cpu(cam_out).numpy(),
            "joints": detach_to_cpu(smplx_out.joints).numpy(),
            "full_body_pose": full_body_pose,
            "jaw_pose": detach_to_cpu(smplx_out.jaw_pose).numpy(),
            "betas": detach_to_cpu(smplx_out.betas).numpy(),
            "expression": detach_to_cpu(smplx_out.expression).numpy(),
            "global_orient": detach_to_cpu(smplx_out.global_orient).numpy(),
            "contacts": detach_to_cpu(contacts).numpy(),
            "bbox_xyxy": bbox_xyxy_np,
            "fps": np.array([fps], dtype=np.int32),
        }
        output_npz_path = Path(cfg.output_dir).parent / f"{video_name}_body_{i}.npz"
        np.savez(str(output_npz_path), **smplx_params_np)
        created_output_files.append(output_npz_path)
        t_smpl_end = time.time()
        if cfg.verbose:
            Log.info(
                f"[SMPL-X] Parameters saved in {output_npz_path}. SMPL-X processing for track {i+1} took {t_smpl_end-t_smpl_start:.4f}s"
            )
    t_total_end = time.time()
    if cfg.verbose:
        Log.info(f"[inference_single_video] Total inference time: {t_total_end-t_total_start:.2f}s")
    # Cleans up the preprocess folder.
    cleanup_temp(cfg)
    torch.cuda.empty_cache()
    gc.collect()
    return created_output_files


def clean_cfg_for_pickling(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return cfg_dict


def preprocess_single_video_wrapper(cfg_dict):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_cfg["gpu_memory_fraction"])
    video_path = cfg_dict["input_video_path"]
    cfg = create_cfg_for_video(
        Path(video_path),
        output_root=cfg_dict["output_dir"],
        static_cam=cfg_dict["static_cam"],
        verbose=cfg_dict["verbose"],
    )
    local_tracker = VideoTracker(cfg_dict["input_root"])
    local_tracker.start_heartbeat()
    try:
        if not local_tracker.is_video_processed(video_path):
            torch.cuda.empty_cache()
            gc.collect()
            preprocess_single_video(cfg)
        else:
            Log.info(f"Video already processed : {video_path}")

        global_hmr4d_model = get_hmr4d_model(cfg)
        global_smplx_model = make_smplx("supermotion")
        loadvitpose = torch.load(cfg.paths.vitpose)
        all_vitpose = loadvitpose["vitpose"]
        flag_by_track = loadvitpose["flagged"]
        created_files = inference_single_video(
            {"cfg": cfg, "data": load_data_dict(cfg), "all_vitpose": all_vitpose, "flag_by_track": flag_by_track},
            global_hmr4d_model,
            global_smplx_model,
        )
        processing_metadata = {
            "processing_time": datetime.datetime.now().isoformat(),
            "status": "success",
            "created_files": [str(f) for f in created_files],
        }
        folder_completed = local_tracker.mark_video_processed(video_path, processing_metadata)
        if cfg.verbose:
            print(f"[DEBUG] folder_completed={folder_completed}")
        if folder_completed:
            merge_body_npz(cfg.output_root)

    except Exception as e:
        tb = traceback.format_exc()
        Log.error(f" Erreur pour {video_path}:\n{tb}")
        local_tracker.mark_video_failed(video_path, tb)
        return None
    finally:
        local_tracker.stop_heartbeat()
        torch.cuda.empty_cache()
        gc.collect()

    return True


# ========== Main funtcion  ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True, help="Root folder containing video_0, video_1, etc.")
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root output folder that will mirror the structure (the .npz files will replace the videos)",
    )
    parser.add_argument("--static_cam", action="store_true", help="If specified, ignore SLAM (DPVO)")
    parser.add_argument("--verbose", action="store_true", help="If specified, display intermediate results")
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="If specified, reprocess all videos even if they've been processed before",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=_cfg["max_frames"],
        help="Maximum duration (in frames) for a video; longer videos will be truncated",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    videotracker = VideoTracker(input_root)
    Log.info(f"Using video tracking folder: {input_root}")

    max_workers = _cfg["max_workers"]
    no_reservation_cycles = 0
    max_no_reservation_cycles = _cfg["max_no_reservation_cycles"]

    all_video_files = sorted(
        [str(p.resolve()) for p in input_root.rglob("*") if p.suffix.lower() in [".mp4", ".avi", ".mov"]]
    )
    remaining_videos = set(all_video_files)

    # important loop
    while True:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            videotracker.cleanup_processed()
            videos_to_process = []
            for video_file in sorted(remaining_videos):
                video_path = Path(video_file)
                if args.force_reprocess or (
                    videotracker.is_video_available(video_path) and not videotracker.is_video_in_progress(video_path)
                ):
                    output_subdir = output_root / video_path.parent.name
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    videos_to_process.append(
                        {"video_file": str(video_path), "output_dir": str(output_subdir.resolve())}
                    )
                else:
                    remaining_videos.discard(str(Path(video_file).resolve()))

            if not videos_to_process:
                Log.info("Toutes les vidéos traitées. Fin.")
                break
            video_paths = [v["video_file"] for v in videos_to_process]
            reserved_videos = videotracker.reserve_videos(video_paths, max_workers, local=True)
            if not reserved_videos:
                no_reservation_cycles += 1
                Log.info(f"No video available, waiting for {_cfg['retry_wait_sec']}s...")
                time.sleep(_cfg["retry_wait_sec"])
                if no_reservation_cycles >= max_no_reservation_cycles:
                    Log.error("No videos available after maximum cycles, exiting.")
                    break
                continue
            else:
                no_reservation_cycles = 0

            # launches the futures on only reserved videos
            batch_infos = [v for v in videos_to_process if v["video_file"] in reserved_videos]
            Log.info(f"Processing batch of {len(batch_infos)} videos")
            futures = []
            future_to_video = {}
            for video_info in batch_infos:
                video_file = Path(video_info["video_file"])
                output_dir = Path(video_info["output_dir"])
                cfg = create_cfg_for_video(
                    video_file,
                    output_root=str(output_dir),
                    static_cam=args.static_cam,
                    verbose=args.verbose,
                    max_frames=args.max_frames,
                )

                # We create a dict because we can't pickle cfg directly :)
                cfg_dict = clean_cfg_for_pickling(cfg)
                cfg_dict["input_video_path"] = str(video_file.resolve())
                cfg_dict["input_root"] = str(input_root.resolve())
                cfg_dict["output_dir"] = str(output_dir.resolve())
                cfg_dict["static_cam"] = args.static_cam
                cfg_dict["verbose"] = args.verbose

                f = executor.submit(preprocess_single_video_wrapper, cfg_dict)
                future_to_video[f] = video_file
                futures.append(f)
                remaining_videos.discard(str(video_file.resolve()))
            for future in futures:
                video_path = future_to_video.get(future, "<unknown>")
                try:
                    result = future.result()
                    if result:
                        Log.info(f"Process sucess for {video_path}")
                    else:
                        Log.warning(f"Process failed or stopped for {video_path}")
                except Exception as e:
                    tb = traceback.format_exc()
                    Log.error(f"Fatal error for {video_path}:\n{tb}")
                    videotracker.mark_video_failed(video_path, tb)
            videotracker.clear_local_reservations()
            torch.cuda.empty_cache()
            gc.collect()
            if args.verbose:
                print_memory_usage()
            else:
                time.sleep(1)
        gc.collect()
        torch.cuda.empty_cache()

        if args.verbose:
            print_memory_usage()

    Log.info("Pipeline done. Success.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()

    try:
        main()
    finally:
        from multiprocessing import active_children

        for p in active_children():
            p.terminate()
        try:
            cleanup = getattr(multiprocessing.resource_tracker, "_cleanup", None)
            if cleanup:
                cleanup()
        except Exception as e:
            Log.warning(f"Multiprocessing cleanup failed: {e}")

    total_time = round(time.time() - start_time, 2)
    print("Body Pipeline executed in:", total_time, "s.")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)
