from __future__ import annotations
import argparse, os, time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import torch
import open3d as o3d
from open3d.visualization import gui, rendering

"""Viewer for HumanML3D 263D .npy files (new_joint_vecs format)

Same controls as debug_visu_anim.py, but for the 263D representation instead of
SMPL-X scenes. That format carries joint positions only, so this renders the
22-joint skeleton -- there is no mesh to show without fitting SMPL back onto it.

Pass several files to overlay them, which is what makes reconstructions
readable: debug_visu_263.py --npy original.npy reconstruction.npy
"""

ap = argparse.ArgumentParser()
ap.add_argument("--npy", type=str, nargs="+", required=True, help="one or more 263D .npy")
ap.add_argument("--mean", type=str, default=None, help="Mean.npy, if the input is normalized")
ap.add_argument("--std", type=str, default=None, help="Std.npy, idem")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
args = ap.parse_args()

JOINTS_NUM = 22

MOTION_COLORS = [
    (0.50, 0.75, 1.00),
    (1.00, 0.60, 0.35),
    (0.45, 0.90, 0.55),
    (1.00, 0.45, 0.55),
    (0.80, 0.55, 1.00),
]

# HumanML3D kinematic chains (t2m_kinematic_chain), flattened into bone pairs.
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
SKELETON_EDGES = np.array(
    [(c[i], c[i + 1]) for c in KINEMATIC_CHAIN for i in range(len(c) - 1)],
    dtype=np.int32,
)

JOINT_RADIUS = 0.035
BONE_RADIUS = 0.022


# SKELETON GEOMETRY
# Lines would be cheaper, but an unlit LineSet neither receives light nor casts
# shadows. Solid spheres and bones give the scene lighting something to work on.
def _align_z(u: np.ndarray) -> np.ndarray:
    """Rotation matrix sending +Z onto the unit vector u (Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(z, u), float(np.dot(z, u))
    s = np.linalg.norm(v)
    if s < 1e-8:  # parallel: identity, or a flip about X for the antiparallel case
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def skeleton_templates():
    """Unit sphere and unit cylinder (+Z, length 1), plus the shared topology.

    Topology never changes across frames, so triangles are built once and only
    vertices are recomputed while playing.
    """
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6)
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=1.0, resolution=8)
    Vs, Ts = np.asarray(sph.vertices), np.asarray(sph.triangles)
    Vc, Tc = np.asarray(cyl.vertices), np.asarray(cyl.triangles)

    tris, off = [], 0
    for _ in range(JOINTS_NUM):
        tris.append(Ts + off)
        off += len(Vs)
    for _ in range(len(SKELETON_EDGES)):
        tris.append(Tc + off)
        off += len(Vc)
    return Vs, Vc, np.concatenate(tris).astype(np.int32)


def skeleton_vertices(joints: np.ndarray, Vs: np.ndarray, Vc: np.ndarray) -> np.ndarray:
    """Vertices of the solid skeleton for one frame of joints (22, 3)."""
    parts = [Vs * JOINT_RADIUS + joints[j] for j in range(JOINTS_NUM)]
    for a, b in SKELETON_EDGES:
        d = joints[b] - joints[a]
        length = float(np.linalg.norm(d))
        if length < 1e-8:  # degenerate bone, park a flat disc at the joint
            parts.append(Vc * [BONE_RADIUS, BONE_RADIUS, 0.0] + joints[a])
            continue
        R = _align_z(d / length)
        parts.append((Vc * [BONE_RADIUS, BONE_RADIUS, length]) @ R.T + (joints[a] + d * 0.5))
    return np.concatenate(parts)


def precompute_normals(verts_frames: List[np.ndarray], tris: np.ndarray):
    """Normals for every frame, reusing a single TriangleMesh."""
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_frames[0]),
        triangles=o3d.utility.Vector3iVector(tris),
    )
    out = []
    for V in verts_frames:
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.compute_vertex_normals()
        out.append(np.asarray(mesh.vertex_normals).copy())
    return out


# 263D -> JOINTS
# Self-contained copy of common.quaternion + utils.motion_process.recover_from_ric,
# so this viewer does not drag in the training package just to decode a file.
def qinv(q):
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q, v):
    """Rotate vectors v by quaternions q, both (*, 4) and (*, 3)."""
    shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(shape)


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    # The first frame is the reference, so velocities integrate from index 1.
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,))
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]  # root height is absolute, not a velocity
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num=JOINTS_NUM):
    data = data.float()
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)


def load_263(path: str, mean, std, start: int, end: int) -> Dict:
    data = np.load(path)
    if data.ndim != 2 or data.shape[-1] != 263:
        raise ValueError(f"{path}: expected (T, 263), got {data.shape}")
    if mean is not None:
        data = data * std + mean
    T_all = data.shape[0]
    s, e = max(0, start), (T_all if end < 0 else min(end, T_all))
    joints = recover_from_ric(torch.from_numpy(data[s:e])).numpy()
    return dict(name=os.path.basename(path), joints=joints, T=joints.shape[0])


# GROUND
def checker_quads(size=12.0, tile=0.5, y=0.0):
    N = max(1, int(round(size / tile)))
    step, half = size / N, size * 0.5
    verts, tris, cols = [], [], []
    vid, z0 = 0, -half
    for iz in range(N):
        x0 = -half
        for ix in range(N):
            verts.extend(
                [
                    (x0, y, z0),
                    (x0 + step, y, z0),
                    (x0, y, z0 + step),
                    (x0 + step, y, z0 + step),
                ]
            )
            tris.extend([(vid, vid + 1, vid + 2), (vid + 1, vid + 3, vid + 2)])
            c = (0.6, 0.6, 0.6) if (ix + iz) % 2 == 0 else (0.50, 0.50, 0.50)
            cols.extend([c] * 4)
            vid += 4
            x0 += step
        z0 += step
    m = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(verts, np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(tris, np.int32)),
    )
    m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cols, np.float64))
    return m


@dataclass
class State:
    i: int = 0
    playing: bool = True
    speed: float = 1.0
    last: float = field(default_factory=time.perf_counter)
    loop: bool = True


class Viewer:
    def __init__(self, motions: List[Dict], fps: int):
        self.motions = motions
        self.idx = 0
        self.show_all = True
        self.shadows_on = False
        self.T = min(m["T"] for m in motions)
        self.target_dt = 1.0 / max(1.0, float(fps))
        self.state = State()
        self._drawn = [(-1, None)] * len(motions)

        # Precompute the solid skeleton once per frame, as debug_visu_anim.py
        # does for its meshes: rebuilding it during playback stutters.
        Vs, Vc, self._tris = skeleton_templates()
        self._verts, self._normals = [], []
        for m in motions:
            print(f"Precomputing '{m['name']}' ({m['T']} frames)")
            vs = [skeleton_vertices(m["joints"][t], Vs, Vc) for t in range(m["T"])]
            self._verts.append(vs)
            self._normals.append(precompute_normals(vs, self._tris))

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("HumanML3D 263D Viewer", 1440, 900)
        em = self.window.theme.font_size
        margin = 0.5 * em

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        # Lit materials, so the Shadows toggle actually changes something.
        self._mats, self._mats_dim, self._meshes = [], [], []
        for i in range(len(motions)):
            r, g, b = MOTION_COLORS[i % len(MOTION_COLORS)]
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            mat.base_color = (r, g, b, 1.0)
            self._mats.append(mat)
            dim = rendering.MaterialRecord()
            dim.shader = "defaultLitTransparency"
            dim.base_color = (r * 0.35, g * 0.35, b * 0.35, 0.85)
            self._mats_dim.append(dim)
            self._meshes.append(
                o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(self._verts[i][0]),
                    triangles=o3d.utility.Vector3iVector(self._tris),
                )
            )

        # recover_from_ric already puts the feet near y=0, but a reconstruction
        # can drift, so sit the floor on the lowest joint of the whole batch --
        # minus the joint radius, or the foot spheres sink into it.
        y_ground = min(float(m["joints"][:, :, 1].min()) for m in motions) - JOINT_RADIUS
        ground = checker_quads(size=14.0, tile=0.5, y=y_ground)
        ground.compute_vertex_normals()
        mat_g = rendering.MaterialRecord()
        mat_g.shader = "defaultLit"
        self.scene_widget.scene.add_geometry("ground", ground, mat_g)
        self.scene_widget.scene.show_skybox(True)
        self.scene_widget.scene.set_background([0.05, 0.05, 0.17, 1.0])
        self._apply_lighting()

        pts = np.concatenate([m["joints"].reshape(-1, 3) for m in motions])
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(pts)
        )
        self.center = bbox.get_center()
        self.eye = self.center + np.array([1.0, 2.5, -4.5])
        self.up = np.array([0.0, 1.0, 0.0])
        offset = self.eye - self.center
        self.radius = np.linalg.norm(offset)
        self.theta = np.arctan2(offset[0], offset[2])
        self.phi = np.arcsin(offset[1] / self.radius)
        self.scene_widget.setup_camera(60.0, bbox, self.center)
        self.scene_widget.look_at(self.center, self.eye, self.up)
        self.scene_widget.set_on_mouse(self.on_mouse)
        self.last_mouse = None

        self.panel = gui.Vert(0, gui.Margins(margin, margin, margin, margin))
        row = gui.Horiz(0.25 * em)
        self.btn_play = gui.Button("Pause")
        b_prev, b_next = gui.Button("< Prev"), gui.Button("Next >")
        b_prev.set_on_clicked(lambda: self.step(-1))
        b_next.set_on_clicked(lambda: self.step(+1))
        self.btn_play.set_on_clicked(self._toggle_play)
        for b in (b_prev, self.btn_play, b_next):
            row.add_child(b)
        self.panel.add_child(row)

        self.label_frame = gui.Label("Frame: 0 / 0")
        self.panel.add_child(self.label_frame)

        self.panel.add_child(gui.Label("Speed"))
        sl = gui.Slider(gui.Slider.DOUBLE)
        sl.set_limits(0.1, 3.0)
        sl.double_value = 1.0
        sl.set_on_value_changed(lambda v: setattr(self.state, "speed", float(v)))
        self.panel.add_child(sl)

        self.cb_all = gui.Checkbox("Show all motions")
        self.cb_all.checked = self.show_all
        self.cb_all.set_on_checked(self._on_show_all)
        self.panel.add_child(self.cb_all)

        self.cb_shadows = gui.Checkbox("Shadows")
        self.cb_shadows.checked = self.shadows_on
        self.cb_shadows.set_on_checked(self._on_toggle_shadows)
        self.panel.add_child(self.cb_shadows)

        self.panel.add_child(gui.Label("Motions"))
        self.label_active = gui.Label(f"Active: {motions[0]['name']}")
        self.panel.add_child(self.label_active)
        for i, m in enumerate(motions):
            btn = gui.Button(f"[{i}] {m['name']} ({m['T']}f)")
            btn.set_on_clicked(lambda i=i: self._set_active(i))
            self.panel.add_child(btn)

        self.window.add_child(self.panel)

        def on_layout(_):
            r = self.window.content_rect
            w = 300
            self.panel.frame = gui.Rect(r.get_right() - w, r.y, w, r.height)
            self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - w, r.height)

        self.window.set_on_layout(on_layout)
        self._redraw(0, force=True)

    def _redraw(self, t: int, force: bool = False):
        sc = self.scene_widget.scene
        frame_t = min(t, self.T - 1)
        visible = set(range(len(self.motions)) if self.show_all else [self.idx])

        for i, m in enumerate(self.motions):
            name = f"skel_{i}"
            is_active = i == self.idx
            ti = min(frame_t, m["T"] - 1)
            if i not in visible:
                if sc.has_geometry(name):
                    sc.remove_geometry(name)
                self._drawn[i] = (-1, None)
                continue
            if not force and self._drawn[i] == (ti, is_active):
                continue
            if sc.has_geometry(name):
                sc.remove_geometry(name)
            mesh = self._meshes[i]
            mesh.vertices = o3d.utility.Vector3dVector(self._verts[i][ti])
            mesh.vertex_normals = o3d.utility.Vector3dVector(self._normals[i][ti])
            mat = self._mats[i] if is_active else self._mats_dim[i]
            sc.add_geometry(name, mesh, mat)
            self._drawn[i] = (ti, is_active)

        self.label_frame.text = f"Frame: {frame_t} / {self.T - 1}"
        self.window.post_redraw()

    def run(self):
        frame_duration = 1.0 / 60.0
        while self.app.run_one_tick():
            t0 = time.perf_counter()
            self._tick()
            remaining = frame_duration - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def _tick(self):
        now = time.perf_counter()
        target = (
            self.target_dt / max(1e-6, self.state.speed) if self.state.playing else 1e9
        )
        if now - self.state.last >= target:
            self.state.last = now
            if self.state.playing:
                self.seek((self.state.i + 1) % self.T)

    def step(self, di: int):
        self.seek((self.state.i + di) % self.T)

    def seek(self, i: int):
        self.state.i = int(i)
        self._redraw(self.state.i)

    def _toggle_play(self):
        self.state.playing = not self.state.playing
        self.btn_play.text = "Pause" if self.state.playing else "Play"
        self.state.last = time.perf_counter()

    def _apply_lighting(self):
        profile = (
            rendering.Open3DScene.LightingProfile.MED_SHADOWS
            if self.shadows_on
            else rendering.Open3DScene.LightingProfile.NO_SHADOWS
        )
        self.scene_widget.scene.set_lighting(
            profile, np.array([-1.0, -1.5, -0.5], dtype=np.float32)
        )

    def _on_toggle_shadows(self, checked: bool):
        self.shadows_on = checked
        self._apply_lighting()
        self.window.post_redraw()

    def _on_show_all(self, checked: bool):
        self.show_all = checked
        self._redraw(self.state.i, force=True)

    def _set_active(self, i: int):
        self.idx = i
        self.label_active.text = f"Active: {self.motions[i]['name']}"
        self._redraw(self.state.i, force=True)

    # CAMERA
    def update_camera(self):
        x = self.radius * np.cos(self.phi) * np.sin(self.theta)
        y = self.radius * np.sin(self.phi)
        z = self.radius * np.cos(self.phi) * np.cos(self.theta)
        self.eye = np.array([x, y, z]) + self.center
        self.scene_widget.look_at(self.center, self.eye, self.up)
        self.scene_widget.force_redraw()

    def on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self.last_mouse = (event.x, event.y)
        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            self.last_mouse = None
        elif event.type == gui.MouseEvent.Type.DRAG and self.last_mouse is not None:
            dx, dy = event.x - self.last_mouse[0], event.y - self.last_mouse[1]
            self.last_mouse = (event.x, event.y)
            if event.is_button_down(gui.MouseButton.LEFT):
                self.orbit(dx, dy)
            elif event.is_button_down(gui.MouseButton.RIGHT):
                self.pan(dx, dy)
        elif event.type == gui.MouseEvent.Type.WHEEL:
            self.zoom(event.wheel_dy)
        return gui.SceneWidget.EventCallbackResult.CONSUMED

    def orbit(self, dx, dy):
        self.theta += dx * 0.01
        self.phi = np.clip(self.phi + dy * 0.01, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        self.update_camera()

    def zoom(self, dy):
        self.radius = max(0.1, self.radius * (1.0 - dy * 0.1))
        self.update_camera()

    def pan(self, dx, dy):
        speed = 0.002 * self.radius
        forward = self.center - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        move = -dx * speed * right + dy * speed * up
        self.center += move
        self.eye += move
        self.update_camera()


if __name__ == "__main__":
    mean = np.load(args.mean) if args.mean else None
    std = np.load(args.std) if args.std else None
    if (mean is None) != (std is None):
        raise SystemExit("--mean and --std go together")
    motions = [load_263(p, mean, std, args.start, args.end) for p in args.npy]
    for m in motions:
        print(f"{m['name']}: {m['T']} frames")
    Viewer(motions, args.fps).run()
