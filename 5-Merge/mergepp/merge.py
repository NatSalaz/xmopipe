import os
import argparse
import yaml
import numpy as np
from pathlib import Path

_cfg = yaml.safe_load(open(Path(__file__).parent.parent.parent / "config.yml"))["merge"]
from scipy.optimize import linear_sum_assignment
import glob
from smoothing import *
from collections import defaultdict
import torch
from shutil import copyfile
from tqdm import tqdm


def compute_cost(
    nose_avg,
    leye_avg,
    reye_avg,
    vitpose,
    return_per_frame: bool = False,
):
    """
    Computes the association cost between face and bodies using mean distances.
    """
    nb_frames = min(nose_avg.shape[0], vitpose.shape[0])

    nose_avg_sel = nose_avg[:nb_frames]
    leye_avg_sel = leye_avg[:nb_frames]
    reye_avg_sel = reye_avg[:nb_frames]
    vitpose_nose = vitpose[:nb_frames, 0, :2]
    vitpose_reye = vitpose[:nb_frames, 1, :2]
    vitpose_leye = vitpose[:nb_frames, 2, :2]

    per_frame_cost = (
        np.linalg.norm(nose_avg_sel - vitpose_nose, axis=1)
        + np.linalg.norm(leye_avg_sel - vitpose_leye, axis=1)
        + np.linalg.norm(reye_avg_sel - vitpose_reye, axis=1)
    )
    avg_cost = float(per_frame_cost.mean())

    return (avg_cost, per_frame_cost) if return_per_frame else avg_cost


def process_all_files(input_body_root, input_face_root, output_root, smoothed):
    """
    Process all .npz files in the input directories, merging face and body data.
    """
    import glob

    # Retrieve .npz files inside subfolders
    body_npz_files = glob.glob(
        os.path.join(input_body_root, "**", "*.npz"), recursive=True
    )
    face_npz_files = glob.glob(
        os.path.join(input_face_root, "**", "*.npz"), recursive=True
    )

    if not body_npz_files or not face_npz_files:
        print("No .npz file found in the given folders.")
        return

    face_map = {}
    for p in face_npz_files:
        base = os.path.basename(p)
        name, ext = os.path.splitext(base)
        parts = name.split("_")
        if len(parts) == 2 and parts[1] == "face":
            face_map[base] = p

    for body_path in tqdm(body_npz_files, desc="File treatment"):
        base = os.path.basename(body_path)
        name, ext = os.path.splitext(base)
        parts = name.split("_")
        if len(parts) != 2 or parts[1] != "body":
            continue
        parent_folder = os.path.basename(os.path.dirname(body_path))
        face_filename = os.path.basename(body_path).replace("body", "face")
        face_path = face_map.get(face_filename)
        if not face_path:
            continue

        try:
            with np.load(body_path, allow_pickle=True) as data:
                if "bodies" not in data:
                    print(f"Skipping (no 'bodies' key): {body_path}")
                    continue
                body_data = data["bodies"].item()
        except Exception as e:
            print(f"Skipping invalid or corrupt file: {body_path} ({e})")
            continue
        try:
            with np.load(face_path, allow_pickle=True) as data:
                if "faces" not in data:
                    print(f"Skipping (no 'faces' key): {face_path}")
                    continue
                face_data = data["faces"].item()
        except Exception as e:
            print(f"Skipping invalid or corrupt file: {face_path} ({e})")
            continue

        for scene_name, scene_bodies in body_data.items():
            scene_faces = face_data.get(scene_name, {})

            body_ids = list(scene_bodies.keys())
            face_ids = list(scene_faces.keys())
            merged_scene = {}

            # Matching using the distance cost matrix of the nose, left eye, and right eye
            cost_matrix = np.zeros((len(face_ids), len(body_ids)), dtype=np.float64)
            for i, face_id in enumerate(face_ids):
                face = scene_faces[face_id]
                nose_avg = np.mean(np.array(face["nose_2d"], dtype=np.float64), axis=1)
                leye_avg = np.mean(np.array(face["leye_2d"], dtype=np.float64), axis=1)
                reye_avg = np.mean(np.array(face["reye_2d"], dtype=np.float64), axis=1)
                for j, body_id in enumerate(body_ids):
                    vitpose = np.array(
                        scene_bodies[body_id]["vitpose"], dtype=np.float64
                    )
                    cost_matrix[i, j] = compute_cost(
                        nose_avg, leye_avg, reye_avg, vitpose
                    )
            # print(f"Cost matrix: {cost_matrix}")
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_body_ids = set()
            pairs = []
            for i, j in zip(row_ind, col_ind):
                face_id = face_ids[i]
                body_id = body_ids[j]
                matched_body_ids.add(body_id)
                pairs.append(
                    (face_id, body_id, scene_faces[face_id], scene_bodies[body_id])
                )

            # Add unmatched bodies
            for body_id in body_ids:
                if body_id not in matched_body_ids:
                    pairs.append((None, body_id, None, scene_bodies[body_id]))
            all_flagged = set()
            for _, _, _, body in pairs:
                flagged = body.get("flagged_frames", [])
                if len(flagged) > 0:
                    all_flagged.update(flagged)
            max_frames = max(body["cam_transl"].shape[0] for _, _, _, body in pairs)
            valid_mask = np.ones(max_frames, dtype=bool)
            valid_mask[list(all_flagged)] = False

            best_len = 0
            best_range = (0, 0)
            current_start = None

            for i, v in enumerate(valid_mask):
                if v and current_start is None:
                    current_start = i
                elif not v and current_start is not None:
                    length = i - current_start
                    if length > best_len:
                        best_len = length
                        best_range = (current_start, i - 1)
                    current_start = None
            if current_start is not None:
                length = max_frames - current_start
                if length > best_len:
                    best_len = length
                    best_range = (current_start, max_frames - 1)

            common_valid_idx = (best_range[0] + best_range[1]) // 2
            print(
                f"[Common frame] Selected index {common_valid_idx} "
                f"(valid range {best_range[0]}–{best_range[1]}, length {best_range[1] - best_range[0] + 1})"
            )

            for face_id, body_id, face, body in pairs:
                N = body["full_body_pose"].shape[0]
                poses = body["full_body_pose"].copy()

                if face is not None:
                    exp = face["exp"]
                    exp_out = np.zeros((N, 50), dtype=exp.dtype)
                    rows, cols = min(N, exp.shape[0]), min(50, exp.shape[1])
                    exp_out[:rows, :cols] = exp[:rows, :cols]
                    jaw = face["jaw"]
                    emo = face["emotions"]
                    emo_conf = face["emotions_conf"]
                    face_bbox = face["bboxes"]
                    face_shape = face["shape"]
                    print("conf", emo_conf.shape)
                    print("face_bbox", face_bbox.shape)
                    print("face_shape", face_shape.shape)
                else:
                    exp_out = np.zeros((N, 50), dtype=np.float32)
                    jaw = np.zeros((N, 3), dtype=poses.dtype)
                    emo = np.full((N,), "unknown", dtype="<U7")
                    jaw = np.zeros((N, 3), dtype=poses.dtype)
                    emo_conf = np.zeros((N, 7), dtype=poses.dtype)
                    face_bbox = np.zeros((N, 4), dtype=poses.dtype)
                    face_shape = np.zeros((N, 300), dtype=poses.dtype)
                # print(N, "=", body["full_body_pose"].shape)
                # print(jaw.shape)
                poses[:, 66:69] = jaw

                # Computes camera translation offset in order to have everything centered
                for idx in range(body["cam_transl"].shape[0]):
                    if (
                        body["flagged_frames"] is not None
                        and idx in body["flagged_frames"]
                    ):
                        continue
                print(
                    "FLAGGED FRAMES ",
                    body_path,
                    "Scene:",
                    scene_name,
                    "==== BODY ",
                    body_id,
                    " ==== ",
                    body.get("flagged_frames", []),
                )
                valid_start, valid_end = best_range
                valid_indices = [
                    i
                    for i in range(valid_start, valid_end + 1)
                    if i not in body.get("flagged_frames", [])
                ]

                if valid_indices:
                    cam_offset = np.mean(body["cam_transl"][valid_indices], axis=0)
                else:
                    cam_offset = body["cam_transl"][common_valid_idx].copy()

                cam_offset[2] *= -1
                trans = body["trans"] - cam_offset
                if smoothed:
                    emo = smoothEmotions(emo, _cfg["emotion_smooth_window"])
                    exp_out = smooth2D_gaussian(exp_out, _cfg["gaussian_window"], _cfg["gaussian_sigma"])
                    poses[:, 66:69] = smooth2D_gaussian(poses[:, 66:69], _cfg["gaussian_window"], _cfg["gaussian_sigma"])

                flag = np.zeros(N, dtype=bool)
                flag_idx = np.array(body.get("flagged_frames", []), dtype=int)
                flag[flag_idx] = True

                merged_scene[body_id] = {
                    "model": "smplx2020",
                    "mocap_framerate": body.get("fps", 30.0),
                    "expressions": exp_out,
                    "trans": trans,
                    "betas": body.get("betas", np.zeros((10,), dtype=np.float32)),
                    "poses": poses,
                    "gender": "neutral",
                    "cam_transl": body["cam_transl"],
                    "emotions": emo,
                    "emotions_conf": emo_conf,
                    "face_bbox_xyxy": face_bbox,
                    "flagged_frames": flag,
                    "bbox_xyxy": body.get(
                        "bbox_xyxy", np.zeros((N, 4), dtype=np.float32)
                    ),
                    "contacts_conf": body.get(
                        "contacts", np.zeros((1, N, 6), dtype=np.float32)
                    ).squeeze(0),
                    "face_shape": face_shape,
                }
                # To see what key are kept or not
                # for bkey in body.keys():
                #    print(f"{bkey},", end="")
                # print("\n + ")
                # if face is not None:
                #    for fkey in face.keys():
                #        print(f"{fkey},", end="")
                # print("\n ===> ")
                # for mskey in merged_scene[body_id].keys():
                #    print(f"{mskey},", end="")
                # print("")

            output_scene_dir = os.path.join(output_root, parent_folder)
            os.makedirs(output_scene_dir, exist_ok=True)

            output_filename = f"{parent_folder}_merged_{scene_name}.npz"
            output_path = os.path.join(output_scene_dir, output_filename)
            np.savez(output_path, **merged_scene)

            # print(f"File saved: {output_path}")
            body_meta = os.path.join(os.path.dirname(body_path), "metadata.txt")
            face_meta = os.path.join(os.path.dirname(face_path), "metadata.txt")
            dest_meta = os.path.join(output_scene_dir, "metadata.txt")

            if os.path.isfile(body_meta):
                copyfile(body_meta, dest_meta)
            elif os.path.isfile(face_meta):
                copyfile(face_meta, dest_meta)
