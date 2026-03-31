import os
import numpy as np
from glob import glob

def merge_face_npz(video_dir, keep=False):
    if not os.path.isdir(video_dir):
        print(f"Invalid directory: {video_dir}")
        return

    video_name = os.path.basename(video_dir)
    video_id = video_name.replace("video_", "")

    npz_files = sorted(glob(os.path.join(video_dir, "*_face_*.npz")))
    expected_keys = []

    faces = {}

    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        try:
            scene_str, _, face_str = filename.replace(".npz", "").split("_")
            scene_key = f"scene_{scene_str}"
            face_key = f"face_{face_str}"
            expected_keys.append((scene_key, face_key))

            data = np.load(npz_path, allow_pickle=True)
            if scene_key not in faces:
                faces[scene_key] = {}
            faces[scene_key][face_key] = {k: data[k] for k in data}
        except Exception as e:
            print(f"Error while processing {filename}: {e}")

    # Save merged npz file
    output_path = os.path.join(video_dir, f"{video_id}_face.npz")
    np.savez_compressed(output_path, faces=faces)

    # Verify the merged content
    verified = True
    try:
        loaded = np.load(output_path, allow_pickle=True)["faces"].item()
        for scene_key, face_key in expected_keys:
            if scene_key not in loaded or face_key not in loaded[scene_key]:
                print(f"Missing: {scene_key}/{face_key}")
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
