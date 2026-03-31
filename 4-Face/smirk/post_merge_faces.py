import os
import json
import argparse
from glob import glob
import numpy as np
from pathlib import Path

### Merges every npz if you forgot to do it before with the input root and npz root.
### Is used in this version of the project to get rid of some "close" npzs and save some inodes

def merge_face_npz(video_dir, keep=False):
    video_name = os.path.basename(video_dir)
    video_id = video_name.replace("video_", "")
    npz_files = sorted(glob(os.path.join(video_dir, "*_face_*.npz")))
    expected_keys = []
    faces = {}
    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        try:
            #Here we assume the file name is "sceneID_face_faceID.npz"
            scene_str, _, face_str = filename.replace(".npz", "").split("_")
            scene_key = f"scene_{scene_str}"
            face_key = f"face_{face_str}"
            expected_keys.append((scene_key, face_key))
            data = np.load(npz_path, allow_pickle=True)
            if scene_key not in faces:
                faces[scene_key] = {}
            faces[scene_key][face_key] = {k: data[k] for k in data}
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    output_path = os.path.join(video_dir, f"{video_id}_face.npz")
    if os.path.exists(output_path):
        print(f"Skip: {output_path} already exists")
        return
    np.savez_compressed(output_path, faces=faces)
    try:
        loaded = np.load(output_path, allow_pickle=True)["faces"].item()
        for scene_key, face_key in expected_keys:
            if scene_key not in loaded or face_key not in loaded[scene_key]:
                print(f"Missing {scene_key}/{face_key} in merged file")
                return
    except Exception as e:
        print(f"Verification error: {e}")
        return
    if not keep:
        for f in npz_files:
            os.remove(f)
        print(f"Merged and cleaned: {video_name}")
    else:
        print(f"Merged (kept originals): {video_name}")
        
        
def process_all(input_root, npz_root, keep=False):
    video_dirs = sorted(glob(os.path.join(input_root, "video_*")))
    for video_path in video_dirs:
        video_id = os.path.basename(video_path)
        print(input_root, "face_tracking", f"tracking_{video_id}.json")
        tracking_path = os.path.join(input_root, "face_tracking", f"tracking_{video_id}.json")
        npz_path = os.path.join(npz_root, video_id)

        if not os.path.exists(tracking_path):
            print(f"Skipping {video_id}: tracking JSON not found")
            continue

        with open(tracking_path) as f:
            tracking_data = json.load(f)

        processed_set = set(tracking_data.get("processed", []))
        video_files = sorted(glob(os.path.join(str(Path(video_path).resolve()), "*.mp4")))
        video_names = [os.path.basename(v) for v in video_files]
        if set(video_files).issubset(processed_set):
            print(f"All videos processed for {video_id}, trying the merging...")
            merge_face_npz(npz_path, keep=keep)
        else:
            print(f"Not all videos processed for {video_id}, skipping merge")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Path to root/video_X folders")
    parser.add_argument("--npz_root", required=True, help="Path to npz_root/")
    parser.add_argument("--keep", action="store_true", help="Keep original npz files after merge")
    args = parser.parse_args()

    process_all(args.input_root, args.npz_root, keep=args.keep)
