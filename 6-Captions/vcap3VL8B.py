import os
import gc
import time
import json
import yaml
import argparse
import traceback
import tempfile
import torch
import cv2
import numpy as np
from pathlib import Path

_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yml"))["captions"]
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from bbox_utils import draw_bboxes_to_temp_video
from video_verif import VideoTracker
import logging
import subprocess
import portalocker
from collections import deque
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
logging.getLogger("decord").setLevel(logging.WARNING)


def get_free_vram(label=""):
    try:
        cpu_total = cpu_free = 0
        with open("/proc/meminfo") as f:
            d = {k: int(v.split()[0]) for k, v in (line.split(":") for line in f)}
        cpu_total = d.get("MemTotal", 0) // 1000
        cpu_free = d.get("MemAvailable", 0) // 1000
        print(f"[CPU:{label}] free={cpu_free}MB / total={cpu_total}MB")
    except Exception:
        pass

    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
        frees = []
        for line in out:
            idx, total, used, free = [int(x.strip()) for x in line.split(",")]
            print(f"[GPU{idx}:{label}] used={used}MB free={free}MB / total={total}MB")
            frees.append(free)
        return frees[0] if frees else 0
    except Exception:
        return 0


def run_inference(
    temp_video_path: Path,
    model,
    processor,
    person_ids: list[str],
    max_tokens: int = 2048,
):
    print(f"Running inference on {temp_video_path}")
    ids_text = ", ".join(person_ids) if person_ids else "none visible"
    prompt_text = f"You are a video analyst. Only describe people who are outlined with visible ID numbers coloured the same way as their outline. Ignore any person without an ID. Use no more than 200 words total. Stick to the structure below. Each section must include only the listed IDs. In your specific case, the visible IDs are: {ids_text}. (This is different from the example values provided below.) <SceneDesc> Briefly describe the environment and general situation. Example: A crowded train station. People are walking or waiting. </SceneDesc> <Action> For each visible ID, describe what the person is doing. Start each sentence with 'A person is...' or 'The person is...'. Example: <1>A person is looking at their phone.</1> <2>The person is sitting on a bench.</2> <5>A person is dragging a suitcase.</5> </Action> <BodyDesc> Describe posture and movement for each ID. Example: <1>Standing upright, arms folded, looking down.</1> <2>Sitting, legs stretched, leaning back.</2> <5>Walking slowly, back slightly hunched.</5> </BodyDesc> <Style> Choose 1 to 3 style words from the list below per person (comma-separated). Stick to the list. Allowed styles: Slow, Fast, Stiff, Flexible, Jerky, Smooth, Broad, Narrow, Soft, Energetic, Hesitant, Confident, Graceful, Ungraceful, Heavy, Light, Modest, Proud, Youthful, Old, Upright, Hunched, Anxious, Calm, Admiration, Approval, Annoyance, Neutral, Gratitude, Disapproval, Amusement, Curiosity, Love, Optimism, Disappointment, Joy, Realisation, Anger, Sadness, Confusion, Caring, Excitement, Surprise, Disgust, Desire, Fear, Remorse, Embarrassment, Nervousness, Relief, Grief Example: <1>Proud, Youthful</1> <2>Anger</2> <5>Excitement, Flexible, Smooth</5> </Style> Reminders: - Use only the IDs provided in your case: {ids_text} - Use short, efficient sentences - All 5 sections are required - Only use style words from the list - Keep total output under 200 words"

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(temp_video_path),
                        "max_pixels": _cfg["max_pixels"],
                        "fps": _cfg["inference_fps"],
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _, video_inputs = process_vision_info(messages)

        if not video_inputs or video_inputs[0].shape[1] == 0:
            print(f"[WARNING] Empty or invalid video input for {temp_video_path.name}")
            return "ERROR"

        inputs = processor(
            text=[text],
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(
                        f"[WARNING] Detected NaN/Inf in input tensor '{k}' for {temp_video_path.name}"
                    )
                    return "ERROR"

        inputs = inputs.to("cuda")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        get_free_vram("pre_generate")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            get_free_vram("post_generate")
            trimmed_ids = [o[len(i) :] for i, o in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return output_text[0] if output_text else "EMPTY"
    except Exception as e:
        print(
            f"[ERROR] Inference failed for {temp_video_path.name}:\n{traceback.format_exc()}"
        )
        return "ERROR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_root", type=str, required=True, help="Root folder containing videos"
    )
    parser.add_argument(
        "--npz_root", type=str, required=True, help="Root folder containing npz files"
    )
    parser.add_argument(
        "--model_path", type=str, default=_cfg["model_path"], help="Path to Qwen VL model"
    )
    parser.add_argument("--max_tokens", type=int, default=_cfg["max_tokens"])
    parser.add_argument("--json_out", type=str, default="video_descriptions.json")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional max number of frames to process per video",
    )
    parser.add_argument("--num_videos", type=int, default=_cfg["num_videos"], help="Reserved videos")
    workers = _cfg["workers"]

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parser.parse_args()
    get_free_vram("startup")

    print("Loading model and processor...")
    torch.cuda.empty_cache()
    gc.collect()
    get_free_vram("pre_load")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    get_free_vram("post_load")
    print(f"Model on device: {model.device}")

    video_root = Path(args.video_root).resolve()
    npz_root = Path(args.npz_root)

    videoTracker = VideoTracker(str(video_root), tracking_file=_cfg["tracking_file"])
    videoTracker.start_heartbeat()

    remaining_videos = deque(str(p.resolve()) for p in video_root.rglob("*.mp4"))

    print(f"{len(remaining_videos)} video(s) to process.")
    json_out_path = Path(args.json_out)

    while remaining_videos:
        videoTracker.cleanup_processed()
        videoTracker.clear_local_reservations()
        videos_to_process = []
        while remaining_videos:
            video_file = remaining_videos.popleft()
            video_path = Path(video_file)
            relative_path = video_path.relative_to(video_root)
            video_id = video_path.stem
            subfolder = relative_path.parent.name
            npz_file = npz_root / subfolder / f"{subfolder}_merged_scene_{video_id}.npz"
            if not npz_file.exists():
                print(f"Skipping {video_path.name}: missing {npz_file.name}")
                continue
            if videoTracker.is_video_available(
                video_path
            ) and not videoTracker.is_video_in_progress(video_path):
                videos_to_process.append(video_path)
                break
        if not videos_to_process:
            print("All video processed or some are unavailable right now.")
            break
        reserved_videos = videoTracker.reserve_videos(
            videos_to_process, max_workers=workers, batch_size=args.num_videos
        )
        if not reserved_videos:
            print(f"No available video, waiting for {_cfg['retry_wait_sec']}s...")
            time.sleep(_cfg["retry_wait_sec"])
            continue

        for video_path in reserved_videos:
            video_path = Path(video_path)
            relative_path = video_path.relative_to(video_root)
            video_id = video_path.stem
            subfolder = relative_path.parent.name
            npz_file = npz_root / subfolder / f"{subfolder}_merged_scene_{video_id}.npz"

            try:
                temp_video, person_ids = draw_bboxes_to_temp_video(
                    video_path, npz_file, max_frames=args.max_frames
                )

                max_wait = _cfg["temp_video_max_wait_sec"]
                elapsed = 0
                while not temp_video.exists() and elapsed < max_wait:
                    time.sleep(0.5)
                    elapsed += 0.5

                if not temp_video.exists():
                    raise FileNotFoundError(f"Temp video not created: {temp_video}")

                time.sleep(1.0)

                cap_verify = cv2.VideoCapture(str(temp_video))
                frame_count = int(cap_verify.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_verify.release()

                if frame_count == 0:
                    raise ValueError(f"Video has 0 frames: {temp_video}")

                output_text = run_inference(
                    temp_video, model, processor, person_ids, max_tokens=args.max_tokens
                )
                json_out_path = (
                    npz_root / subfolder / f"description_videos_{subfolder}.json"
                )
                json_out_path.parent.mkdir(parents=True, exist_ok=True)
                if not json_out_path.exists():
                    with open(json_out_path, "w", encoding="utf-8") as f:
                        json.dump({}, f)

                with portalocker.Lock(json_out_path, mode="r+", timeout=10) as f:
                    try:
                        f.seek(0)
                        current_data = json.load(f)
                    except json.JSONDecodeError:
                        current_data = {}
                    json_key = f"{subfolder}/{video_id}"
                    current_data[json_key] = output_text
                    temp_dir = json_out_path.parent
                    with tempfile.NamedTemporaryFile(
                        "w", dir=str(temp_dir), delete=False
                    ) as tf:
                        json.dump(current_data, tf, indent=2, ensure_ascii=False)
                        tf.flush()
                        os.fsync(tf.fileno())
                        temp_path = Path(tf.name)
                os.replace(temp_path, json_out_path)
                videoTracker.mark_video_processed(str(video_path))
                print(f"Saved result of {video_path.name}")
            except Exception:
                print(f"Error processing {video_path.name}:\n{traceback.format_exc()}")
                videoTracker.mark_video_failed(str(video_path), traceback.format_exc())
            finally:
                try:
                    temp_video.unlink()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
        videoTracker.clear_local_reservations()
    print(
        "\nAll done. Results saved to individual description_videos.json files per subfolder."
    )


if __name__ == "__main__":
    print("GPU count visible:", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    main()
