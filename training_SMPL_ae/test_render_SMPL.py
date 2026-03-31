from __future__ import annotations
import argparse
import numpy as np
import torch
import smplx
import cv2
import open3d as o3d
from open3d.visualization import rendering
from edit_gui.IK import apply_ik
from utils.motion_process import recover_from_ric


# -------------------------------------------------------------------
# SKELETON DEFINITIONS
# -------------------------------------------------------------------

HML_PARENTS = {
    0: -1,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 0,
    7: 6,
    8: 7,
    9: 8,
    10: 0,
    11: 10,
    12: 11,
    13: 12,
    14: 3,
    15: 14,
    16: 15,
    17: 3,
    18: 17,
    19: 18,
    20: 9,
    21: 13,
}

SMPLX_BODY_JOINTS = [
    (1,  1,  0),
    (2,  2,  1),
    (3,  3,  2),
    (4,  4,  3),
    (5,  5,  4),
    (6,  6,  0),
    (7,  7,  6),
    (8,  8,  7),
    (9,  9,  8),
    (10, 10, 0),
    (11, 11, 10),
    (12, 12, 11),
    (13, 13, 12),
    (14, 14, 3),
    (15, 15, 14),
    (16, 16, 15),
    (17, 17, 3),
    (18, 18, 17),
    (19, 19, 18),
    (20, 20, 9),
    (21, 21, 13),
]

FABRIK_CHAINS = {
    "left_leg":  [6, 7, 8, 9],
    "right_leg": [10, 11, 12, 13],
    "left_arm":  [14, 15, 16],
    "right_arm": [17, 18, 19],
    "spine":     [0, 1, 2, 3, 4, 5],
}


# -------------------------------------------------------------------
# MATH
# -------------------------------------------------------------------

def to_tensor_f32(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    return x.float().to(device)


def rotation_from_vectors(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def matrix_to_axis_angle(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2 * np.sin(angle))
    return axis * angle


def fabrik(chain, target, lengths, iters=100):
    joints = chain.copy()
    root = joints[0].copy()
    for _ in range(iters):
        joints[-1] = target
        for i in reversed(range(len(joints) - 1)):
            d = joints[i] - joints[i + 1]
            d /= np.linalg.norm(d) + 1e-8
            joints[i] = joints[i + 1] + d * lengths[i]
        joints[0] = root
        for i in range(len(joints) - 1):
            d = joints[i + 1] - joints[i]
            d /= np.linalg.norm(d) + 1e-8
            joints[i + 1] = joints[i] + d * lengths[i]
    return joints


# -------------------------------------------------------------------
# IK
# -------------------------------------------------------------------

def apply_fabrik(joints, rest_joints):
    joints = joints.copy()
    for chain_idx in FABRIK_CHAINS.values():
        chain = joints[chain_idx]
        target = chain[-1]
        lengths = [
            np.linalg.norm(rest_joints[chain_idx[i + 1]] - rest_joints[chain_idx[i]])
            for i in range(len(chain_idx) - 1)
        ]
        new_chain = fabrik(chain, target, lengths)
        for i, j in enumerate(chain_idx):
            joints[j] = new_chain[i]
    return joints


def joints_to_smplx_pose(joints, rest_joints):
    pose = []
    for _, j, p in SMPLX_BODY_JOINTS:
        R = rotation_from_vectors(
            rest_joints[j] - rest_joints[p],
            joints[j] - joints[p],
        )
        pose.append(matrix_to_axis_angle(R))
    return np.concatenate(pose, axis=0)


def compute_root_orient(joints):
    left = joints[6]
    right = joints[10]
    fwd = right - left
    yaw = np.arctan2(fwd[2], fwd[0])
    return np.array([0.0, yaw, 0.0], dtype=np.float32)


# -------------------------------------------------------------------
# SMPL-X
# -------------------------------------------------------------------

class SMPLXSequence:
    def __init__(self, model_path, device):
        self.device = torch.device(device)
        self.model = smplx.create(
            model_path,
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            ext="npz",
        ).to(self.device).eval()

        self.faces = self.model.faces.astype(np.int32)
        self.zero = torch.zeros(1, 45, device=self.device)

        self.rest_joints = (
            self.model(
                body_pose=torch.zeros(1, 63, device=self.device),
                betas=torch.zeros(1, 10, device=self.device),
            )
            .joints[0, :22]
            .cpu().detach()
            .numpy()
            .astype(np.float32)
        )

    @torch.no_grad()
    def forward(self, joints_seq):
        verts = []
        for joints in joints_seq:
            joints_fabrik = apply_fabrik(joints, self.rest_joints)
            body_pose = joints_to_smplx_pose(joints_fabrik, self.rest_joints)
            global_orient = compute_root_orient(joints_fabrik)
            transl = joints_fabrik[0]

            out = self.model(
                global_orient=to_tensor_f32(global_orient, self.device).unsqueeze(0),
                body_pose=to_tensor_f32(body_pose, self.device).unsqueeze(0),
                transl=to_tensor_f32(transl, self.device).unsqueeze(0),
                betas=torch.zeros(1, 10, device=self.device),
                left_hand_pose=self.zero,
                right_hand_pose=self.zero,
            )
            verts.append(out.vertices[0].cpu().numpy())
        return np.stack(verts), self.faces


# -------------------------------------------------------------------
# RENDER
# -------------------------------------------------------------------

class Renderer:
    def __init__(self, w=1280, h=720):
        self.renderer = rendering.OffscreenRenderer(w, h)
        self.scene = self.renderer.scene
        self.scene.set_background([1, 1, 1, 1])
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultLit"
        self.w, self.h = w, h

    def setup_camera(self, verts):
        bbox = o3d.geometry.AxisAlignedBoundingBox(verts.min(0), verts.max(0))
        c = bbox.get_center()
        eye = c + np.array([3, 2, 3])
        self.scene.camera.look_at(c, eye, [0, 1, 0])
        self.scene.camera.set_projection(
            60.0, self.w / self.h, 0.1, 100.0,
            rendering.Camera.FovType.Vertical
        )

    def render(self, verts, faces):
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        self.scene.clear_geometry()
        self.scene.add_geometry("smplx", mesh, self.mat)
        return np.asarray(self.renderer.render_to_image())


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--smplx", default="body_models")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"

    motion = np.load(args.npy)
    joints = recover_from_ric(
        torch.from_numpy(motion).float(), joints_num=22
    ).cpu().numpy().astype(np.float32)
    joints[..., 2] *= -1.0

    smpl = SMPLXSequence(args.smplx, device)
    verts_seq, faces = smpl.forward(joints)

    renderer = Renderer()
    renderer.setup_camera(verts_seq[0])

    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (renderer.w, renderer.h),
    )

    for v in verts_seq:
        img = renderer.render(v, faces)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()


if __name__ == "__main__":
    main()
