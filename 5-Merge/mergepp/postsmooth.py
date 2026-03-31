from __future__ import annotations

import argparse
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Tuple

_cfg = yaml.safe_load(open(Path(__file__).parent.parent.parent / "config.yml"))["merge"]

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from remove_flagged import trim_flagged_borders
from smplx import SMPLX


# Utils
def to_int(x: Any) -> int:
    if isinstance(x, np.ndarray):
        x = x.flatten()[0]
    try:
        return int(x)
    except (TypeError, ValueError):
        return int(float(x))


def _continuous_rotvec(rot: R) -> np.ndarray:
    q = rot.as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return R.from_quat(q).as_rotvec()


# 6D to axis-angle
def sixd_to_axis_angle(r6: np.ndarray) -> np.ndarray:
    r6_t = torch.as_tensor(r6, dtype=torch.float32)
    a1, a2 = r6_t[..., :3], r6_t[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(
        a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1
    )
    b3 = torch.cross(b1, b2, dim=-1)
    mats = torch.stack([b1, b2, b3], dim=-2).cpu().numpy().reshape(-1, 3, 3)
    return R.from_matrix(mats).as_rotvec().reshape(*r6.shape[:-1], 3)


# Resampling
def resample_30fps(root_aa: np.ndarray, fps_in: int):
    t_in = np.arange(len(root_aa)) / float(fps_in)
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1 / float(_cfg["target_fps"]))
    return t_out


def resample_rotvec_slerp(
    rotvec: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray
) -> np.ndarray:
    r_src = R.from_rotvec(rotvec)
    r_slerp = Slerp(t_src, r_src)(t_dst)
    return _continuous_rotvec(r_slerp)


def resample_nearest(
    arr: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray
) -> np.ndarray:
    arr = np.asarray(arr)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]

    idx = np.searchsorted(t_src, t_dst, side="left")
    idx = np.clip(idx, 0, len(t_src) - 1)
    idx_left = np.clip(idx - 1, 0, len(t_src) - 1)
    choose_left = np.abs(t_dst - t_src[idx_left]) <= np.abs(t_src[idx] - t_dst)
    idx = np.where(choose_left, idx_left, idx)

    out = arr[idx]
    return out.squeeze() if squeeze else out


def resample_numeric(
    arr: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray
) -> np.ndarray:
    arr = np.asarray(arr)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]

    interp = interp1d(t_src, arr, axis=0, bounds_error=False, fill_value="extrapolate")
    out = interp(t_dst)
    return out.squeeze() if squeeze else out


def resample_pose_slerp(
    aa_src: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray
) -> np.ndarray:
    """Interpolate pose (T, 165) with quaternion continuity"""
    Nsrc, J = aa_src.shape[0], 55
    aa_src = aa_src.reshape(Nsrc, J, 3)

    quat = R.from_rotvec(aa_src.reshape(-1, 3)).as_quat().reshape(Nsrc, J, 4)

    for j in range(J):
        for i in range(1, Nsrc):
            if np.dot(quat[i - 1, j], quat[i, j]) < 0:
                quat[i, j] = -quat[i, j]

    aa_dst = np.empty((len(t_dst), J, 3), dtype=np.float32)
    for j in range(J):
        R_interp = Slerp(t_src, R.from_quat(quat[:, j]))(t_dst)
        aa_dst[:, j] = _continuous_rotvec(R_interp)

    return aa_dst.reshape(len(t_dst), 165)


# In some occurences of our work we had 6D pose representations so it should be obsolete to use the 6D rep; anyway this function converts them to axis-angle.
def split_pose(full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dim = full.shape[1]
    if dim == 330:
        aa = sixd_to_axis_angle(full.reshape(-1, 55, 6)).reshape(-1, 165)
    elif dim == 165:
        aa = full
    else:
        raise ValueError(f"Unexpected pose dimension {dim}")
    return aa[:, :3], aa[:, 3:]


# Motion metrics more or less the ones from Go To Zero (arXiv:2507.07095)
def delta_theta(root_aa: np.ndarray) -> np.ndarray:
    Rm = R.from_rotvec(root_aa).as_matrix()
    rel = Rm[1:] @ Rm[:-1].transpose(0, 2, 1)
    dtheta = np.linalg.norm(R.from_matrix(rel).as_rotvec(), axis=1)
    return np.concatenate([[0.0], dtheta])


def jerk_metric(joints: np.ndarray, fps_in: int) -> np.ndarray:
    dt = 1.0 / float(fps_in)
    v = np.gradient(joints, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    return np.linalg.norm(j, axis=-1).mean(axis=1)


# SMPL-X forward
def smplx_joints(
    body_pose_aa: np.ndarray,
    transl: np.ndarray,
    smplx_model: SMPLX,
    betas: np.ndarray,
    expr: np.ndarray,
):
    device = next(smplx_model.parameters()).device
    with torch.no_grad():
        out = smplx_model(
            global_orient=torch.tensor(body_pose_aa[:, :3], device=device).float(),
            body_pose=torch.tensor(body_pose_aa[:, 3:66], device=device).float(),
            jaw_pose=torch.tensor(body_pose_aa[:, 66:69], device=device).float(),
            leye_pose=torch.tensor(body_pose_aa[:, 69:72], device=device).float(),
            reye_pose=torch.tensor(body_pose_aa[:, 72:75], device=device).float(),
            left_hand_pose=torch.tensor(body_pose_aa[:, 75:120], device=device).float(),
            right_hand_pose=torch.tensor(
                body_pose_aa[:, 120:165], device=device
            ).float(),
            transl=torch.tensor(transl, device=device, dtype=torch.float32),
            betas=torch.tensor(betas, device=device, dtype=torch.float32),
            expression=torch.tensor(expr, device=device, dtype=torch.float32),
        )
    return out.joints.cpu().numpy()


# Main processing
def process_npz(npz_path: str, smplx_model: SMPLX, out_dir: str):
    basename = os.path.splitext(os.path.basename(npz_path))[0]
    data = np.load(npz_path, allow_pickle=True)

    new_data: Dict[str, Dict[str, Any]] = {}
    body_keys = sorted([k for k in data.files if k.startswith("body_")])
    base_offset = None

    TH_DTHETA = _cfg["motion_th_dtheta"]
    TH_JERK = _cfg["motion_th_jerk"]

    for key in body_keys:
        body: Dict[str, Any] = data[key].item()

        full_pose = body["poses"]
        betas_raw = body["betas"]
        expr_raw = body.get("expressions")
        transl = body["trans"]
        cam_transl = body.get("cam_transl")
        bbox = body.get("bbox_xyxy")
        emotions = body.get("emotions")
        flags = body.get("flagged_frames")
        contacts_conf = body.get("contacts_conf")
        emotions_conf = body.get("emotions_conf")
        face_bbox = body.get("face_bbox_xyxy")
        face_shape = body.get("face_shape")

        if "mocap_framerate" in body:
            fps_in = to_int(body["mocap_framerate"])
        elif "fps" in body:
            fps_in = to_int(body["fps"])
        else:
            fps_in = _cfg["fallback_fps"]

        aa_full = full_pose
        root_aa, _ = split_pose(aa_full)

        B = smplx_model.num_betas
        E = smplx_model.num_expression_coeffs
        if betas_raw.ndim == 1:
            betas_raw = betas_raw[None, :]
        if betas_raw.shape[0] != len(full_pose):
            betas_raw = np.repeat(betas_raw[:, :B], len(full_pose), axis=0)

        betas = betas_raw
        expr = np.zeros((len(full_pose), E)) if expr_raw is None else expr_raw[:, :E]
        if (
            np.isnan(full_pose).any()
            or np.isnan(transl).any()
            or (expr_raw is not None and np.isnan(expr_raw).any())
        ):
            print(f"/!!!/ {key}: contains NaNs, skipped")
            continue

        joints = smplx_joints(full_pose, transl, smplx_model, betas, expr)
        dtheta = delta_theta(root_aa)
        jerk = jerk_metric(joints, fps_in)
        mean_dtheta = dtheta.mean()
        mean_jerk = jerk.mean()

        if mean_dtheta < TH_DTHETA and mean_jerk < TH_JERK:
            print(
                f"{key}: no motion detected (Δθ={mean_dtheta:.4f}, jerk={mean_jerk:.4e}), skip"
            )
            continue
        # print(f"{key}: Δθ={mean_dtheta:.4f}, jerk={mean_jerk:.4e}")

        t30 = resample_30fps(root_aa, fps_in)
        t_keep = np.arange(len(aa_full)) / float(fps_in)

        pose30 = resample_pose_slerp(aa_full, t_keep, t30)
        transl30 = resample_numeric(transl, t_keep, t30)

        if key == "body_0" and base_offset is None:
            base_offset = transl30[0].copy()
            y_min = joints[..., 1].min()
            # print(y_min, " ground at")
            base_offset[1] += y_min  # puts y=0 at feet level

        expr30 = (
            resample_numeric(expr_raw, t_keep, t30).astype(np.float32)
            if expr_raw is not None
            else None
        )
        cam30 = (
            resample_numeric(cam_transl, t_keep, t30).astype(np.float32)
            if cam_transl is not None
            else None
        )
        bbox30 = (
            resample_numeric(bbox, t_keep, t30).astype(np.float32)
            if bbox is not None
            else None
        )
        emotions30 = (
            resample_nearest(emotions, t_keep, t30).astype(emotions.dtype)
            if emotions is not None
            else None
        )
        flags30 = (
            resample_nearest(flags.astype(bool), t_keep, t30).astype(bool)
            if flags is not None
            else None
        )
        contacts_conf30 = (
            resample_numeric(contacts_conf, t_keep, t30).astype(np.float32)
            if contacts_conf is not None
            else None
        )
        emotions_conf30 = (
            resample_numeric(emotions_conf, t_keep, t30).astype(np.float32)
            if emotions_conf is not None
            else None
        )
        face_bbox30 = (
            resample_numeric(face_bbox, t_keep, t30).astype(np.float32)
            if face_bbox is not None
            else None
        )
        face_shape30 = (
            resample_numeric(face_shape, t_keep, t30).astype(np.float32)
            if face_shape is not None
            else None
        )
        if base_offset is not None:
            transl30 = transl30 - base_offset
            if cam30 is not None:
                cam30 = cam30 - base_offset

        start, stop = (
            trim_flagged_borders(flags30, int(fps_in / 2))
            if flags30 is not None
            else (0, len(pose30))
        )

        if start > 0 or stop < len(pose30):
            pose30 = pose30[start:stop]
            transl30 = transl30[start:stop]
            expr30 = expr30[start:stop] if expr30 is not None else None
            cam30 = cam30[start:stop] if cam30 is not None else None
            bbox30 = bbox30[start:stop] if bbox30 is not None else None
            emotions30 = emotions30[start:stop] if emotions30 is not None else None
            flags30 = flags30[start:stop] if flags30 is not None else None
            contacts_conf30 = (
                contacts_conf30[start:stop] if contacts_conf30 is not None else None
            )
            emotions_conf30 = (
                emotions_conf30[start:stop] if emotions_conf30 is not None else None
            )
            face_bbox30 = face_bbox30[start:stop] if face_bbox30 is not None else None
            face_shape30 = (
                face_shape30[start:stop] if face_shape30 is not None else None
            )
        if pose30.shape[0] == 0:
            print(f"{key} trimmed entirely, skipped")
            continue

        new_body = {
            k: v for k, v in body.items() if k not in {"poses_6d", "mocap_framerate"}
        }
        new_body.update(
            {
                "poses": pose30.astype(np.float32),
                "trans": transl30.astype(np.float32),
                "fps": _cfg["target_fps"],
                "original_fps": fps_in,
                "start": start,
                "stop": stop,
            }
        )

        betas30 = np.tile(betas_raw[0], (pose30.shape[0], 1))
        new_body["betas"] = betas30

        if expr30 is not None:
            new_body["expressions"] = expr30
        if cam30 is not None:
            new_body["cam_transl"] = cam30
        if bbox30 is not None:
            new_body["bbox_xyxy"] = bbox30
        if emotions30 is not None:
            new_body["emotions"] = emotions30
        if flags30 is not None:
            new_body["flagged_frames"] = flags30
        if contacts_conf30 is not None:
            new_body["contacts_conf"] = contacts_conf30
        if emotions_conf30 is not None:
            new_body["emotions_conf"] = emotions_conf30
        if face_bbox30 is not None:
            new_body["face_bbox_xyxy"] = face_bbox30
        if face_shape30 is not None:
            new_body["face_shape"] = face_shape30
        new_data[key] = new_body

    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"{basename}.npz")
    print(f"{len(new_data)} bodies processed => {out_npz}")
    if len(new_data) != 0:
        np.savez(out_npz, **new_data)
    else:
        print(out_npz, " not saved, 0 body to process.")


def main():
    ap = argparse.ArgumentParser(
        description="Process .npz files to resample to 30 fps and center scene around 0,0,0"
    )
    ap.add_argument(
        "--npz-folder", required=True, help="Path to folder containing .npz files"
    )
    ap.add_argument(
        "--smplx-model", default=_cfg["smplx_model_dir"], help="SMPL-X models folder"
    )
    ap.add_argument("--gender", default=_cfg["smplx_gender"], help="SMPL-X gender model")
    ap.add_argument("--output", default=_cfg["output_ppmerged_dir"], help="Output directory")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=_cfg["device"], help="Device")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    smplx_model = SMPLX(
        model_path=args.smplx_model, gender=args.gender, use_pca=False
    ).to(device)

    for root, _, files in os.walk(args.npz_folder):
        rel_path = os.path.relpath(root, args.npz_folder)
        out_dir = os.path.join(args.output, rel_path)
        for f in files:
            if f.endswith(".npz") and not f.endswith("NEUTRAL_2020.npz"):
                npz_path = os.path.join(root, f)
                process_npz(npz_path, smplx_model, out_dir)
            if f.startswith("description_videos_video_") and f.endswith(".json"):
                src_json = os.path.join(root, f)
                os.makedirs(out_dir, exist_ok=True)
                os.system(f'cp "{src_json}" "{out_dir}/"')
                print(f"{f} copied to {out_dir}")
        metadata_path = os.path.join(root, "metadata.txt")
        if os.path.isfile(metadata_path):
            os.makedirs(out_dir, exist_ok=True)
            os.system(f'cp "{metadata_path}" "{out_dir}/"')
            print(f"metadata.txt copied to {out_dir}")


if __name__ == "__main__":
    main()
