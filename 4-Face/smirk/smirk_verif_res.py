#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch multi-person FLAME+Emotion pipeline with inference time display.
Each folder (e.g. video_0) in --input_root must contain:
  - metadata.txt
  - videos
The output replicates this structure by generating NPZ files (one per face).
This script uses the tracking.py module for face tracking.
"""
import cProfile, pstats
from video_verif import VideoTracker

import gc
import os
import yaml
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path

_cfg = yaml.safe_load(open(Path(__file__).parent.parent.parent / "config.yml"))["face"]
import torch.nn.functional as F
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms
from ResEmoteNet import ResEmoteNet
from merge_faces import merge_face_npz
from skimage.transform import estimate_transform, warp
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from tracking import BetterFaceTracker
from src.FLAME.FLAME import FLAME
from src.smirk_encoder import SmirkEncoder
import traceback

# FLAME Constants
JAW_DIM = 3
POSE_DIM = 3
EXP_DIM = 50
NECK_DIM = 3
EYE_POSE_DIM = 6
EYELID_DIM = 2
SHAPE_DIM = 300
EMOTIONS_DIM = 7

# emotion classes, keep in mind that these are PER-FRAME EMOTIONS
class_names = ["happy", "surprise", "sad", "anger", "disgust", "fear", "neutral"]


# ========== Utility Functions ==========
def upscale_image(image, scale=2):
    if not isinstance(image, np.ndarray):
        image = image.cpu().numpy() if hasattr(image, "cpu") else np.array(image)
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)


def crop_face(image, landmarks, scale=1.4, image_size=224):
    left, right = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    top, bottom = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform("similarity", src_pts, dst_pts)


def run_mediapipe_landmarker(face_crop, face_landmarker):
    if face_crop is None or face_crop.size == 0:
        return None
    start = time.time()
    h, w = face_crop.shape[:2]
    image_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = face_landmarker.detect(mp_image)
    elapsed = time.time() - start
    run_mediapipe_landmarker.elapsed = elapsed
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    lm_np = np.zeros((len(lm), 3), dtype=np.float32)
    for i, point in enumerate(lm):
        lm_np[i, 0] = point.x * w
        lm_np[i, 1] = point.y * h
        lm_np[i, 2] = point.z
    return lm_np


def extract_face_embeddings_batch(face_crops, embedding_model, device):
    processed = []
    for crop in face_crops:
        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (160, 160))
        tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).float()
        processed.append(tensor)
    batch_tensor = torch.cat(processed, dim=0).to(device)
    batch_tensor = (batch_tensor / 255.0 - 0.5) * 2.0
    with torch.no_grad():
        embeddings = embedding_model(batch_tensor)
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
    return embeddings


def create_empty_face_params(face_id):
    return {
        "id": face_id,
        "bboxes": np.zeros((4,), dtype=np.float32),
        "nose_2d": np.zeros((3, 2), dtype=np.float32),
        "leye_2d": np.zeros((5, 2), dtype=np.float32),
        "reye_2d": np.zeros((5, 2), dtype=np.float32),
        "jaw": np.zeros((JAW_DIM,), dtype=np.float32),
        "pose": np.zeros((POSE_DIM,), dtype=np.float32),
        "exp": np.zeros((EXP_DIM,), dtype=np.float32),
        "neck": np.zeros((NECK_DIM,), dtype=np.float32),
        "eye_pose": np.zeros((EYE_POSE_DIM,), dtype=np.float32),
        "eyelid": np.zeros((EYELID_DIM,), dtype=np.float32),
        "emotions": "unknown",
        "shape": np.zeros((SHAPE_DIM,), dtype=np.float32),
        "emotions_conf": np.zeros((EMOTIONS_DIM,), dtype=np.float32),
    }


def run_emotion_single(emotion_model, face_crop, device):
    """
    Predicts emotion on a crop
    Parameters:
      - emotion_model: the model.
      - face_crop: the image (numpy array, BGR) with the cropped face.
      - device: on which device we want to preidct.

    returns:
      Dictionnary with label and class probabilities
    """
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = emotion_model(img_tensor)
    probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    label = class_names[scores.argmax()]
    return {"label": label, "probs": scores}


def run_emotion_pytorch_batch(emotion_model, crops):
    results = []
    device = next(emotion_model.parameters()).device
    # Parallelize inference for each crop using threads
    with ThreadPoolExecutor(max_workers=_cfg["emotion_threads"]) as executor:
        futures = [
            executor.submit(run_emotion_single, emotion_model, crop, device)
            for crop in crops
        ]
        for future in futures:
            results.append(future.result())
    return results


class VideoDataset(Dataset):
    def __init__(self, video_path, max_frames=None):
        self.frames = []
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames is not None and count >= max_frames):
                break
            self.frames.append(frame)
            count += 1
        cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


# ========== Video Processing ==========
def process_video(
    video_path,
    output_dir,
    pipeline,
    device,
    input_image_size=224,
    crop=False,
    verbose=True,
    batch_size=8,
    max_frames=500,
):
    print(f"[Start] Processing {video_path}")
    if verbose:
        print(
            f"[MEMORY] Current GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )
        print(
            f"[MEMORY] Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
        )
    video_time = time.time()
    dataset = VideoDataset(video_path, max_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_cfg["dataloader_workers"],
        pin_memory=True,
        prefetch_factor=_cfg["dataloader_prefetch_factor"],
        persistent_workers=True,
    )
    face_tracker = pipeline["face_tracker"]
    person_params = {}
    frame_idx = 0
    scale = _cfg["upscale_factor"]  # upscaling for YOLO
    timing_stats = {
        "yolo": 0.0,
        "mediapipe": 0.0,
        "smirk": 0.0,
        "flame": 0.0,
        "emotion": 0.0,
        "embed": 0.0,
        "warp": 0.0,
    }
    counts = {
        "yolo": 0,
        "mediapipe": 0,
        "smirk": 0,
        "flame": 0,
        "emotion": 0,
        "embed": 0,
        "warp": 0,
    }
    emotion_requests = []  # (face_id, index, crop)
    smirk_inputs = []  #  contain the tensors (crop in RGB)
    smirk_meta = []  #  contain (face_id, face_landmarks) to reconstruct parameters

    num_workers = min(_cfg["dataloader_workers"], os.cpu_count())

    if verbose:
        print(
            f"Video {video_path} loaded in {round(time.time()-video_time,2)}s. Let's process."
        )

    with ThreadPoolExecutor(max_workers=num_workers) as mp_executor:
        for batch in dataloader:
            batch = list(batch)
            upscaled_batch = [upscale_image(frame, scale=scale) for frame in batch]
            start_yolo = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    yolo_results = pipeline["model_yolo"](
                        upscaled_batch, conf=_cfg["yolo_face_confidence"], verbose=False
                    )
            timing_stats["yolo"] += time.time() - start_yolo
            counts["yolo"] += len(upscaled_batch)
            batch_boxes = []
            embedding_crops = []
            mapping = []

            for i, frame in enumerate(batch):
                upscaled = upscaled_batch[i]
                result = yolo_results[i]
                boxes = []
                if hasattr(result, "boxes") and result.boxes is not None:
                    for b in result.boxes:
                        coords = b.xyxy[0].cpu().numpy().astype(int)
                        h_up, w_up = upscaled.shape[:2]
                        x1_up, y1_up = max(0, coords[0]), max(0, coords[1])
                        x2_up, y2_up = min(w_up, coords[2]), min(h_up, coords[3])
                        if x2_up - x1_up <= 0 or y2_up - y1_up <= 0:
                            continue
                        # face crop to use smirk and resemotenet on it
                        face_crop_up = upscaled[y1_up:y2_up, x1_up:x2_up].copy()
                        detection_index = len(boxes)
                        boxes.append(
                            (x1_up, y1_up, x2_up, y2_up, float(b.conf[0]), None, None)
                        )
                        if face_crop_up.size:
                            mapping.append((i, detection_index, len(embedding_crops)))
                            embedding_crops.append(face_crop_up)
                batch_boxes.append(boxes)
            if embedding_crops:
                start_embed = time.time()
                embeddings = extract_face_embeddings_batch(
                    embedding_crops, pipeline["embedding_model"], device
                )
                timing_stats["embed"] += time.time() - start_embed
                counts["embed"] += len(embedding_crops)
                for frame_idx_in_batch, detection_index, emb_index in mapping:
                    boxes = batch_boxes[frame_idx_in_batch]
                    box = boxes[detection_index]
                    updated_box = (
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        box[4],
                        embeddings[emb_index],
                        box[6],
                    )
                    boxes[detection_index] = updated_box
            for i, frame in enumerate(batch):
                boxes = batch_boxes[i]
                tracked_faces = face_tracker.update(boxes, dt=1.0)
                for face_id, box_up in tracked_faces:
                    x1_up, y1_up, x2_up, y2_up = map(int, box_up)
                    # original coordinates
                    x1, y1 = x1_up // scale, y1_up // scale
                    x2, y2 = x2_up // scale, y2_up // scale
                    h_frame, w_frame = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_frame, x2), min(h_frame, y2)
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        param_dict = create_empty_face_params(face_id)
                        if face_id not in person_params:
                            person_params[face_id] = {
                                key: [
                                    np.zeros((dim,), dtype=np.float32)
                                    for _ in range(frame_idx)
                                ]
                                for key, dim in zip(
                                    [
                                        "bboxes",
                                        "nose_2d",
                                        "leye_2d",
                                        "reye_2d",
                                        "jaw",
                                        "pose",
                                        "exp",
                                        "neck",
                                        "eye_pose",
                                        "eyelid",
                                        "shape",
                                        "emotions",
                                        "emotions_conf",
                                    ],
                                    [
                                        4,
                                        3,
                                        5,
                                        5,
                                        JAW_DIM,
                                        POSE_DIM,
                                        EXP_DIM,
                                        NECK_DIM,
                                        EYE_POSE_DIM,
                                        EYELID_DIM,
                                        SHAPE_DIM,
                                        1,
                                        EMOTIONS_DIM,
                                    ],
                                )
                            }
                        for key in person_params[face_id]:
                            person_params[face_id][key].append(param_dict[key])
                        continue
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

                    # MediaPipe
                    future = mp_executor.submit(
                        run_mediapipe_landmarker,
                        upscaled[y1_up:y2_up, x1_up:x2_up].copy(),
                        pipeline["face_landmarker"],
                    )
                    face_landmarks = future.result()
                    timing_stats["mediapipe"] += run_mediapipe_landmarker.elapsed
                    counts["mediapipe"] += 1

                    if face_landmarks is None or face_landmarks.shape[0] < 478:
                        if face_id not in person_params:
                            person_params[face_id] = {
                                key: [
                                    np.zeros((dim,), dtype=np.float32)
                                    for _ in range(frame_idx)
                                ]
                                for key, dim in zip(
                                    [
                                        "bboxes",
                                        "nose_2d",
                                        "leye_2d",
                                        "reye_2d",
                                        "jaw",
                                        "pose",
                                        "exp",
                                        "neck",
                                        "eye_pose",
                                        "eyelid",
                                        "shape",
                                        "emotions",
                                        "emotions_conf",
                                    ],
                                    [
                                        4,
                                        3,
                                        5,
                                        5,
                                        JAW_DIM,
                                        POSE_DIM,
                                        EXP_DIM,
                                        NECK_DIM,
                                        EYE_POSE_DIM,
                                        EYELID_DIM,
                                        SHAPE_DIM,
                                        1,
                                        EMOTIONS_DIM,
                                    ],
                                )
                            }
                        param_dict = create_empty_face_params(face_id)
                        for key in person_params[face_id]:
                            person_params[face_id][key].append(param_dict[key])
                        continue
                    face_landmarks[:, 0] = (face_landmarks[:, 0] + x1_up) / 2.0
                    face_landmarks[:, 1] = (face_landmarks[:, 1] + y1_up) / 2.0

                    frame_np = (
                        frame if isinstance(frame, np.ndarray) else frame.cpu().numpy()
                    )
                    if crop:
                        time_warp = time.time()
                        tform = crop_face(
                            frame_np,
                            face_landmarks[..., :2],
                            scale=_cfg["crop_scale"],
                            image_size=input_image_size,
                        )
                        M = tform.params[:2, :]
                        cropped = cv2.warpAffine(
                            frame_np,
                            M,
                            (input_image_size, input_image_size),
                            flags=cv2.INTER_LINEAR,
                        )
                        timing_stats["warp"] += time.time() - time_warp
                        counts["warp"] += 1
                    else:
                        cropped = cv2.resize(
                            frame_np, (input_image_size, input_image_size)
                        )
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    cropped_tensor = (
                        torch.tensor(cropped_rgb)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                        / 255.0
                    )
                    emotion_crop = upscaled[y1_up:y2_up, x1_up:x2_up].copy()
                    if face_id not in person_params:
                        person_params[face_id] = {
                            key: [
                                np.zeros((dim,), dtype=np.float32)
                                for _ in range(frame_idx)
                            ]
                            for key, dim in zip(
                                [
                                    "bboxes",
                                    "nose_2d",
                                    "leye_2d",
                                    "reye_2d",
                                    "jaw",
                                    "pose",
                                    "exp",
                                    "neck",
                                    "eye_pose",
                                    "eyelid",
                                    "shape",
                                    "emotions",
                                    "emotions_conf",
                                ],
                                [
                                    4,
                                    3,
                                    5,
                                    5,
                                    JAW_DIM,
                                    POSE_DIM,
                                    EXP_DIM,
                                    NECK_DIM,
                                    EYE_POSE_DIM,
                                    EYELID_DIM,
                                    SHAPE_DIM,
                                    1,
                                    EMOTIONS_DIM,
                                ],
                            )
                        }
                    person_params[face_id]["emotions"].append(None)
                    person_params[face_id]["emotions_conf"].append(None)
                    smirk_inputs.append(cropped_tensor)
                    smirk_meta.append((face_id, face_landmarks, bbox))
                    current_index = len(person_params[face_id]["emotions"]) - 1
                    emotion_requests.append((face_id, current_index, emotion_crop))
                tracked_ids = {face_id for face_id, _ in tracked_faces}
                for missing_id in set(person_params.keys()) - tracked_ids:
                    param_dict = create_empty_face_params(missing_id)
                    for key in person_params[missing_id]:
                        person_params[missing_id][key].append(param_dict[key])
                frame_idx += 1

            # ========== Batch processing SMIRK+FLAME ==========
            if smirk_inputs:
                start_smirk = time.time()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        smirk_batch = torch.cat(smirk_inputs, dim=0)
                        smirk_outputs = pipeline["smirk_encoder"](smirk_batch)
                timing_stats["smirk"] += time.time() - start_smirk
                counts["smirk"] += len(smirk_inputs)

                start_flame = time.time()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        flame_outputs = pipeline["flame"](smirk_outputs)
                timing_stats["flame"] += time.time() - start_flame
                counts["flame"] += len(smirk_inputs)
                for j, meta in enumerate(smirk_meta):
                    face_id, face_landmarks, bbox = meta
                    nose = np.array(
                        [
                            face_landmarks[1][:2],
                            face_landmarks[4][:2],
                            face_landmarks[5][:2],
                        ]
                    )
                    left_eye = np.array(
                        [face_landmarks[i][:2] for i in range(468, 473)]
                    )
                    right_eye = np.array(
                        [face_landmarks[i][:2] for i in range(473, 478)]
                    )
                    jaw = flame_outputs["jaw"][j].cpu().detach().numpy().squeeze()
                    pose = flame_outputs["pose"][j].cpu().detach().numpy().squeeze()
                    exp = (
                        flame_outputs["expressions"][j].cpu().detach().numpy().squeeze()
                    )
                    neck = flame_outputs["neck"][j].cpu().detach().numpy().squeeze()
                    eye_pose = (
                        flame_outputs["eye_pose"][j].cpu().detach().numpy().squeeze()
                    )
                    shape = (
                        smirk_outputs["shape_params"][j]
                        .cpu()
                        .detach()
                        .numpy()
                        .squeeze()
                    )
                    eyelid = flame_outputs["eyelid"][j].cpu().detach().numpy().squeeze()
                    param_dict = {
                        "id": face_id,
                        "bboxes": bbox,
                        "nose_2d": nose,
                        "leye_2d": left_eye,
                        "reye_2d": right_eye,
                        "jaw": jaw,
                        "pose": pose,
                        "exp": exp,
                        "neck": neck,
                        "eye_pose": eye_pose,
                        "eyelid": eyelid,
                        "shape": shape,
                    }
                    for key in [
                        "bboxes",
                        "nose_2d",
                        "leye_2d",
                        "reye_2d",
                        "jaw",
                        "pose",
                        "exp",
                        "neck",
                        "eye_pose",
                        "eyelid",
                        "shape",
                    ]:
                        person_params[face_id][key].append(param_dict[key])
                smirk_inputs = []
                smirk_meta = []

            # ========== Processing emotion requests for the batch ==========
            if emotion_requests:
                start_emotion = time.time()
                crops = [req[2] for req in emotion_requests]
                emotion_results = run_emotion_pytorch_batch(
                    pipeline["emotion_model"], crops
                )
                timing_stats["emotion"] += time.time() - start_emotion
                counts["emotion"] += len(emotion_requests)
                for (face_id, idx, _), result in zip(emotion_requests, emotion_results):
                    label = class_names[np.argmax(result["probs"])]
                    person_params[face_id]["emotions"][idx] = label
                    person_params[face_id]["emotions_conf"][idx] = result["probs"]
                emotion_requests = []
        gc.collect()
        torch.cuda.empty_cache()
    if verbose:
        print("Total inference time and average per model:")
        for key in timing_stats:
            avg = timing_stats[key] / counts[key] if counts[key] > 0 else 0
            print(
                f" - {key}: total = {timing_stats[key]:.3f}s over {counts[key]} calls, average = {avg:.3f}s/call"
            )

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    for face_id, params in person_params.items():
        if any(np.array(item).shape != (3, 2) for item in params["nose_2d"]):
            continue
        if any(np.array(item).shape != (5, 2) for item in params["leye_2d"]):
            continue
        if any(np.array(item).shape != (5, 2) for item in params["reye_2d"]):
            continue
        npz_filename = f"{video_name}_face_{face_id-1}.npz"
        npz_path = os.path.join(output_dir, npz_filename)
        np.savez(
            npz_path,
            bboxes=np.array(params["bboxes"]),
            nose_2d=np.array(params["nose_2d"], dtype=object),
            leye_2d=np.array(params["leye_2d"], dtype=object),
            reye_2d=np.array(params["reye_2d"], dtype=object),
            jaw=np.array(params["jaw"]),
            pose=np.array(params["pose"]),
            exp=np.array(params["exp"]),
            neck=np.array(params["neck"]),
            eye_pose=np.array(params["eye_pose"]),
            eyelid=np.array(params["eyelid"]),
            emotions=np.array(params["emotions"]),
            emotions_conf=np.array(params["emotions_conf"]),
            shape=np.array(params["shape"]),
        )
        if verbose:
            print(f"Saved: {npz_filename}")
    face_tracker.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Finished processing {video_path} in {round(time.time()-video_time,2)}s.")
    print(
        f"[MEMORY] GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    print(
        f"[MEMORY] Peak GPU memory during run: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB\n"
    )


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(
        description="Batch FLAME+Emotion pipeline with tracking and statistics"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Input root folder (containing video_*/ with metadata and subfolders)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root folder (structure replicated with NPZ files)",
    )
    parser.add_argument(
        "--smirk_checkpoint",
        type=str,
        default=_cfg["smirk_checkpoint"],
        help="Checkpoint for the SMIRK encoder",
    )
    parser.add_argument(
        "--device", type=str, default=_cfg["device"], help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=_cfg["batch_size"], help="Batch size for video processing"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=_cfg["max_frames"],
        help="Maximum number of frames to process per video",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_cfg["gpu_memory_fraction"])

    start_time = time.time()
    device = args.device
    input_image_size = _cfg["input_image_size"]
    input_root = Path(args.input_root).resolve()
    videoTracker = VideoTracker(str(input_root), tracking_file=_cfg["tracking_file"])
    videoTracker.start_heartbeat()

    # emotion model
    pytorch_emotion_model = ResEmoteNet()
    checkpoint = torch.load(_cfg["emotion_checkpoint"], map_location=device)
    pytorch_emotion_model.load_state_dict(checkpoint["model_state_dict"])
    pytorch_emotion_model.eval()

    # load models onto GPU and initialize the pipeline
    embedding_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    smirk_encoder = SmirkEncoder().to(device)
    checkpoint = torch.load(args.smirk_checkpoint, map_location=device)
    checkpoint_encoder = {
        k.replace("smirk_encoder.", ""): v
        for k, v in checkpoint.items()
        if "smirk_encoder" in k
    }
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()
    flame = FLAME().to(device)
    model_yolo = YOLO(_cfg["yolo_model"]).to(device)
    base_options = python.BaseOptions(
        model_asset_path=_cfg["mediapipe_model"],
        delegate=python.BaseOptions.Delegate.GPU,
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=_cfg["mediapipe_num_faces"],
        min_face_detection_confidence=_cfg["mediapipe_min_detection_confidence"],
        min_face_presence_confidence=_cfg["mediapipe_min_presence_confidence"],
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    face_tracker = BetterFaceTracker(
        max_miss=_cfg["face_tracker_max_miss"],
        iou_threshold=_cfg["face_tracker_iou_threshold"],
        appearance_weight=_cfg["face_tracker_appearance_weight"],
    )

    pipeline = {
        "embedding_model": embedding_model,
        "smirk_encoder": smirk_encoder,
        "flame": flame,
        "model_yolo": model_yolo,
        "face_landmarker": face_landmarker,
        "emotion_model": pytorch_emotion_model,
        "face_tracker": face_tracker,
    }

    all_video_paths = sorted(
        [
            str(p.resolve())
            for p in Path(args.input_root).rglob("*")
            if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
        ]
    )

    for video_path in all_video_paths:
        if not videoTracker.is_video_available(video_path):
            if args.verbose:
                print(f"Video already processed or in progress: {video_path}")
            continue
        if not videoTracker.mark_video_in_progress(video_path):
            if args.verbose:
                print(f"Could not mark video as in progress: {video_path}")
            continue
        rel_path = Path(video_path).relative_to(input_root)
        output_subdir = Path(args.output_root) / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        metadata_src = Path(args.input_root) / rel_path.parent / "metadata.txt"
        if metadata_src.exists():
            shutil.copy(metadata_src, output_subdir)
        try:
            torch.cuda.empty_cache()
            gc.collect()
            if args.verbose:
                print(f"Processing {video_path} ...")
            process_video(
                video_path,
                str(output_subdir),
                pipeline,
                device,
                input_image_size,
                crop=True,
                batch_size=args.batch_size,
                verbose=args.verbose,
                max_frames=args.max_frames,
            )

            metadata = {"processing_time": round(time.time() - start_time, 2)}
            folder_completed = videoTracker.mark_video_processed(video_path, metadata)
            if args.verbose:
                print(f"[DEBUG] folder_completed={folder_completed}")
            if folder_completed:
                merge_face_npz(output_subdir)
                print("Merging npzs")
        except Exception as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"[OOM] Out of memory for {video_path}, skipping")
                videoTracker.mark_video_failed(video_path, f"OOM: {str(e)}")
            else:
                tb = traceback.format_exc()
                videoTracker.mark_video_failed(video_path, str(e))
                if args.verbose:
                    print(f"Error processing {video_path}: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    videoTracker.clear_local_reservations()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()
    main()
    print("Face Pipeline executed in:", round(time.time() - start_time, 2), "s.")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)  # displays the top 20 most time-consuming functions
