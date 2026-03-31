import os
import numpy as np
from glob import glob

""" 
Merges individual body npz files into a single npz file per video directory.
"""


def merge_body_npz(video_dir, keep=False):
    if not os.path.isdir(video_dir):
        print(f"Invalid directory: {video_dir}")
        return

    video_name = os.path.basename(video_dir)
    video_id = video_name.replace("video_", "")

    npz_files = sorted(glob(os.path.join(video_dir, "*_body_*.npz")))
    expected_keys = []

    bodies = {}

    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        try:
            scene_str, _, body_str = filename.replace(".npz", "").split("_")
            scene_key = f"scene_{scene_str}"
            body_key = f"body_{body_str}"
            expected_keys.append((scene_key, body_key))

            data = np.load(npz_path, allow_pickle=True)
            if scene_key not in bodies:
                bodies[scene_key] = {}
            bodies[scene_key][body_key] = {k: data[k] for k in data}
        except Exception as e:
            print(f"Error while processing {filename}: {e}")

    # Save merged npz file
    output_path = os.path.join(video_dir, f"{video_id}_body.npz")
    np.savez_compressed(output_path, bodies=bodies)

    # Verify the merged content
    verified = True
    try:
        loaded = np.load(output_path, allow_pickle=True)["bodies"].item()
        for scene_key, body_key in expected_keys:
            if scene_key not in loaded or body_key not in loaded[scene_key]:
                print(f"Missing: {scene_key}/{body_key}")
                verified = False
    except Exception as e:
        print(f"Verification error: {e}")
        verified = False

    # Delete or keep original files
    if verified:
        if not keep:
            for f in npz_files:
                os.remove(f)
            print(f"Merge successful and original files deleted for {video_name}")
        else:
            print(f"Merge successful for {video_name}, original files kept")
    else:
        print(f"Incomplete merge for {video_name}, original files retained")
