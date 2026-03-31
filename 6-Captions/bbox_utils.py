import os
import gc
import math
import contextlib
from pathlib import Path

import cv2
import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt


_load = torch.load


def torch_load_patch(*args, **kwargs):
    kwargs["weights_only"] = False
    return _load(*args, **kwargs)


torch.load = torch_load_patch


def calculate_mask_iou(mask, bbox):
    if mask is None or len(mask.shape) < 2:
        return 0.0
    x1, y1, x2, y2 = map(int, bbox)
    bbox_mask = np.zeros_like(mask, dtype=np.uint8)
    bbox_mask[y1:y2, x1:x2] = 1
    intersection = np.logical_and(mask, bbox_mask).sum()
    union = np.logical_or(mask, bbox_mask).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def draw_bboxes_to_temp_video(
    video_path: Path,
    npz_path: Path,
    model_path="FastSAM-x.pt",
    model=None,
    max_frames: int | None = None,
    batch_size: int = 64,
    iou_threshold: float = 0.30,
    min_contour_area: int = 200,
    imgsz: int = 512,
    margin: int = 5,
) -> tuple[Path, list[str]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_ctx = (
        torch.cuda.amp.autocast if device == "cuda" else contextlib.nullcontext
    )
    if model is None:
        model = FastSAM(model_path)
    if device == "cuda":
        model.model.half()
    model.model.to(device)

    data = np.load(npz_path, allow_pickle=True)
    bodies = {}
    person_ids = []
    for k in data.files:
        if k.startswith("body_"):
            body_data = data[k].item()
            if "bbox_xyxy" in body_data:
                pid = k.replace("body_", "")
                bodies[k] = body_data["bbox_xyxy"]
                person_ids.append(pid)
    if not bodies:
        raise ValueError(f"No valid 'bbox_xyxy' found in {npz_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # borne commune #frames
    min_len = num_frames_video
    for b in bodies.values():
        min_len = min(min_len, b.shape[0])
    if max_frames:
        min_len = min(min_len, max_frames)

    temp_dir = "./tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_video_path = Path(temp_dir) / f"temp_overlay_{os.getpid()}.mp4"
    out = cv2.VideoWriter(
        str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    PALETTE = [
        (0, 0, 255),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (255, 128, 0),
        (0, 128, 255),
        (128, 0, 255),
        (128, 255, 0),
        (255, 0, 128),
        (0, 255, 128),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3

    # auto-adaptative batches, some videos have a lot of information so it just does not seem to work with high batch sizes
    cur_bs = max(1, batch_size)

    with torch.inference_mode():
        frame_idx_global = 0
        while frame_idx_global < min_len:
            # start frame of the batch, when we rewind we come back from here
            start_frame_idx = frame_idx_global
            batch_frames = []
            frames_to_read = min(cur_bs, min_len - frame_idx_global)
            for _ in range(frames_to_read):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                batch_frames.append(frame)

            if not batch_frames:
                break

            while True:
                try:
                    with autocast_ctx():
                        everything_results = model(
                            batch_frames,
                            device=device,
                            retina_masks=True,
                            verbose=False,
                            imgsz=imgsz,
                        )
                    break
                except RuntimeError as e:
                    if (
                        "CUDA" in str(e).upper()
                        and "OUT OF MEMORY" in str(e).upper()
                        and cur_bs > 1
                    ):
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        gc.collect()
                        print(
                            "Dividing batch_size. => current batch size:",
                            cur_bs,
                            "=>",
                            (cur_bs // 2),
                        )
                        cur_bs = max(1, cur_bs // 2)
                        batch_frames = batch_frames[:cur_bs]
                        #  rewinding to the batch starting frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
                        batch_frames = []
                        frames_to_read = min(cur_bs, min_len - start_frame_idx)
                        for _ in range(frames_to_read):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if frame.dtype != np.uint8:
                                frame = frame.astype(np.uint8)
                            batch_frames.append(frame)
                        continue
                    else:
                        raise

            while True:
                bad_prompt = False
                prompts_cache: list[FastSAMPrompt | None] = []
                try:
                    for i, frame in enumerate(batch_frames):
                        try:
                            prompt_process = FastSAMPrompt(
                                frame, [everything_results[i]], device=device
                            )
                        except Exception:
                            prompt_process = None
                        if prompt_process is None:
                            bad_prompt = True
                            prompts_cache = []
                            break
                        prompts_cache.append(prompt_process)
                except Exception:
                    bad_prompt = True
                    prompts_cache = []

                if bad_prompt and cur_bs > 1:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()

                    print(
                        "Dividing batch_size. => current batch size:",
                        cur_bs,
                        "=>",
                        (cur_bs // 2),
                    )
                    cur_bs = max(1, cur_bs // 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
                    batch_frames = []
                    frames_to_read = min(cur_bs, min_len - start_frame_idx)
                    for _ in range(frames_to_read):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        batch_frames.append(frame)
                    with autocast_ctx():
                        everything_results = model(
                            batch_frames,
                            device=device,
                            retina_masks=True,
                            verbose=False,
                            imgsz=imgsz,
                        )
                    continue
                else:
                    break

            for i, frame in enumerate(batch_frames):
                fidx = frame_idx_global + i

                prompt_process = None
                if i < len(prompts_cache):
                    prompt_process = prompts_cache[i]
                if prompt_process is None:
                    try:
                        prompt_process = FastSAMPrompt(
                            frame, [everything_results[i]], device=device
                        )
                    except Exception as e:
                        print(f"FastSAMPrompt error frame {fidx}: {e}")
                        out.write(frame)
                        continue
                    if prompt_process is None:
                        # cur_bs == 1 and nothing worked => raw frame ...
                        print(
                            f"FastSAMPrompt is None at frame {fidx}, writing raw frame."
                        )
                        out.write(frame)
                        continue

                for body_key, bboxes in bodies.items():
                    person_id = body_key.replace("body_", "")
                    try:
                        bbox = bboxes[fidx]
                    except Exception:
                        continue
                    if bbox is None or np.any(np.isnan(bbox)):
                        continue

                    x1, y1, x2, y2 = map(int, bbox)
                    if (
                        x1 >= x2
                        or y1 >= y2
                        or x1 < 0
                        or y1 < 0
                        or x2 > width
                        or y2 > height
                    ):
                        continue

                    x1m = max(0, x1 - margin)
                    y1m = max(0, y1 - margin)
                    x2m = min(width, x2 + margin)
                    y2m = min(height, y2 + margin)

                    try:
                        masks = prompt_process.box_prompt([x1m, y1m, x2m, y2m])
                    except Exception as e:
                        print(f"box_prompt error frame {fidx}, person {person_id}: {e}")
                        continue
                    if masks is None or len(masks) == 0:
                        continue

                    best_mask, best_iou = None, 0.0
                    for m in masks:
                        mu = m.astype(np.uint8)
                        iou = calculate_mask_iou(mu, (x1, y1, x2, y2))
                        if iou > best_iou:
                            best_iou, best_mask = iou, mu
                    if best_mask is None or best_iou < iou_threshold:
                        continue

                    # morpho clean
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    contours = [
                        c for c in contours if cv2.contourArea(c) > min_contour_area
                    ]
                    if not contours:
                        continue

                    mean_top_x, top_y = None, None
                    for c in contours:
                        min_y = int(c[:, 0, 1].min())
                        mask_top = (c[:, 0, 1] >= min_y) & (c[:, 0, 1] <= min_y + 10)
                        if np.any(mask_top):
                            mean_top_x = int(c[:, 0, 0][mask_top].mean())
                            top_y = min_y
                        else:
                            mean_top_x = int(c[:, 0, 0].mean())
                            top_y = min_y

                    color = (
                        PALETTE[int(person_id) % len(PALETTE)]
                        if person_id.isdigit()
                        else (0, 255, 0)
                    )
                    cv2.drawContours(frame, contours, -1, color, 2)

                    if contours and mean_top_x is not None:
                        min_y = min(cnt[:, 0, 1].min() for cnt in contours)
                        center_x = mean_top_x
                        label = person_id
                        (tw, th), _ = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )
                        text_x = int(center_x) - tw // 2
                        text_y = int(min_y) - 10
                        if text_y - th < 0:
                            text_y = int(min_y) + th + 10
                        cv2.rectangle(
                            frame,
                            (text_x - 2, text_y - th - 2),
                            (text_x + tw + 2, text_y + 4),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            frame,
                            label,
                            (text_x, text_y),
                            font,
                            font_scale,
                            color,
                            thickness,
                            lineType=cv2.LINE_AA,
                        )

                out.write(frame)

            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            frame_idx_global += len(batch_frames)

    cap.release()
    out.release()
    person_ids.sort(key=lambda x: int(x) if x.isdigit() else x)
    return temp_video_path, person_ids
