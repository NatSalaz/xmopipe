import argparse
import os
import shutil
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg

_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yml"))["download"]


def process_video(task):
    """
    Processes a single video:
      - Detects scenes
      - Filters out scenes shorter than the specified minimum duration
      - Splits the video using ffmpeg
      - Appends metadata text with scene timecodes
      - Optionally deletes the source video after processing it
    Returns:
        tuple: (folder_name, filename, error_message or None, metadata_text)
    """
    folder, filename, video_path, output_subdir, min_duration, delete_after = task

    # Open the video and initialize scene manager
    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector())
    manager.detect_scenes(video)
    scenes = manager.get_scene_list(start_in_scene=True)

    # Keep only scenes longer than minimum duration
    valid_scenes = [
        (start, end)
        for start, end in scenes
        if end.get_seconds() - start.get_seconds() >= min_duration
    ]

    if not valid_scenes:
        return (folder, filename, f"No scenes longer than {min_duration}s found.", "")

    # Prepare output file template
    base_name = os.path.splitext(filename)[0]
    template = f"{base_name}-$SCENE_NUMBER.mp4"

    # Perform splitting
    result = split_video_ffmpeg(
        input_video_path=video_path,
        scene_list=valid_scenes,
        output_dir=output_subdir,
        output_file_template=template,
        show_output=True,
        arg_override="-map 0:v:0 -map 0:a? -map 0:s? -c copy",
    )

    if result != 0:
        return (folder, filename, f"Error splitting video (code {result}).", "")

    # Optionally delete original video
    if delete_after:
        try:
            os.remove(video_path)
        except Exception as e:
            print(f"Warning: could not delete {video_path}: {e}")

    # metadata text for the processed video
    lines = [f"\n{filename}:"]
    for idx, (start, end) in enumerate(valid_scenes, start=1):
        lines.append(f"Scene {idx}: {start.get_timecode()} - {end.get_timecode()}")
    metadata_text = "\n".join(lines) + "\n"

    return (folder, filename, None, metadata_text)


def main():
    parser = argparse.ArgumentParser(
        description="Video scene detection and splitting script"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not remove the entire input directory after processing.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=_cfg["min_scene_duration"],
        help="Minimum scene duration (in seconds) to keep (default from config.yml).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=_cfg["scraped_dir"],
        help="Input folder of videos with their metadatas as harvested as done with YTscrap.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_cfg["cut_dir"],
        help="Output folder of videos with scenes cut with their metadatas",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    # Prepare tasks for all videos in subfolders
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Scanning folder: {folder}")
        sub_out = os.path.join(output_dir, folder)
        os.makedirs(sub_out, exist_ok=True)

        # Copy metadata
        orig_meta = os.path.join(folder_path, "metadata.txt")
        dest_meta = os.path.join(sub_out, "metadata.txt")
        if os.path.exists(orig_meta):
            shutil.copy(orig_meta, dest_meta)

        # Queue video files
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                path = os.path.join(folder_path, fname)
                tasks.append(
                    (folder, fname, path, sub_out, args.min_duration, not args.keep)
                )

    results = []
    # Process videos
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_video, t): t for t in tasks}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing videos"
        ):
            try:
                results.append(future.result())
            except Exception as e:
                task = futures[future]
                print(f"Error with {task[1]} in {task[0]}: {e}")

    # Write aggregated metadata
    metadata_collection = {}
    for folder, fname, error, meta in results:
        if error:
            print(f"{fname} in {folder}: {error}")
        else:
            metadata_collection.setdefault(folder, []).append(meta)

    for folder, metas in metadata_collection.items():
        meta_file = os.path.join(output_dir, folder, "metadata.txt")
        with open(meta_file, "a") as f:
            for m in metas:
                f.write(m)
    # Remove processed video folders in input_dir if they only contain metadata.txt or are empty
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        remaining = [f for f in os.listdir(folder_path) if f != "metadata.txt"]
        if not remaining:
            try:
                shutil.rmtree(folder_path)
                print(f"Removed empty folder '{folder_path}'")
            except Exception as e:
                print(f"Warning: could not remove folder {folder_path}: {e}")
    # Remove entire input directory unless --keep is set
    if not args.keep:
        print(f"Removing input directory '{input_dir}'...")
        shutil.rmtree(input_dir)
        print("Input directory removed.")
    else:
        print("Input directory retained.")


if __name__ == "__main__":
    main()
