from __future__ import annotations
import argparse, time
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from smplx import SMPLX
import open3d as o3d
from open3d.visualization import gui, rendering

"""Viewer for NPZ scene files (multi-character, SMPLX 2020 format)
"""

# ARGUMENTS
ap = argparse.ArgumentParser()
ap.add_argument("--npz", type=str, required=True)
ap.add_argument(
    "--model_path",
    type=str,
    default="data/smplx_models/smplx",
)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--pca_hands", action="store_true")
ap.add_argument("--hands_flat", action="store_true")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
args = ap.parse_args()

use_cuda = (not args.cpu) and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Device: {device}")

sl = slice
POSE_IDX = {
    "global_orient": sl(0, 3),
    "body_pose": sl(3, 66),
    "left_hand_pose": sl(66, 111),
    "right_hand_pose": sl(111, 156),
    "jaw_pose": sl(156, 159),
}

CHAR_COLORS = [
    (0.50, 0.75, 1.00),
    (1.00, 0.60, 0.35),
    (0.45, 0.90, 0.55),
    (1.00, 0.45, 0.55),
    (0.80, 0.55, 1.00),
]

SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
]
SKELETON_EDGES_NP = np.array(SKELETON_EDGES, dtype=np.int32)


# LOAD NPZ
def load_npz(npz_path: str, start: int, end: int) -> List[Dict]:
    raw = np.load(npz_path, allow_pickle=True)
    keys = sorted([k for k in raw.keys() if k.startswith("body_")])
    if not keys:
        raise ValueError(f"No 'body_*' keys in {npz_path}. Keys: {list(raw.keys())}")
    print(f"Found {len(keys)} character(s): {keys}")
    characters = []
    for key in keys:
        body = raw[key].item()
        T_all = body["poses"].shape[0]
        s = max(0, start)
        e = T_all if end < 0 else min(end, T_all)
        fps = int(body.get("fps", 30))
        gender = body.get("gender", "neutral")
        poses = torch.from_numpy(body["poses"][s:e]).float()
        trans = torch.from_numpy(body["trans"][s:e]).float()
        betas = torch.from_numpy(body["betas"][s:e]).float()
        expressions = torch.from_numpy(body["expressions"][s:e]).float()
        characters.append(
            dict(
                name=key,
                gender=gender,
                fps=fps,
                poses=poses,
                trans=trans,
                betas=betas,
                expressions=expressions,
                T=poses.shape[0],
            )
        )
    return characters


# SMPLX MODELS
_smplx_cache: Dict[str, SMPLX] = {}


def get_model(gender: str) -> SMPLX:
    if gender not in _smplx_cache:
        import os

        model_path = args.model_path
        if os.path.isdir(model_path):
            candidate = os.path.join(model_path, f"SMPLX_{gender.upper()}_2020.npz")
            if os.path.exists(candidate):
                model_path = candidate
                print(f"Using SMPLX-2020 file: {candidate}")
        m = (
            SMPLX(
                model_path=model_path,
                gender=gender,
                use_pca=args.pca_hands,
                flat_hand_mean=args.hands_flat,
                num_expression_coeffs=50,
                use_face_contour=True,
            )
            .to(device)
            .eval()
        )
        _smplx_cache[gender] = m
        print(f"Loaded SMPLX (gender={gender})")
    return _smplx_cache[gender]


def get_faces(gender: str) -> np.ndarray:
    return get_model(gender).faces.astype(np.int32)


@torch.no_grad()
def smplx_vertices(char: Dict, t: int) -> Tuple[np.ndarray, np.ndarray]:
    model = get_model(char["gender"])
    pose = char["poses"][t].to(device)
    expr = char["expressions"][t].to(device)
    beta = char["betas"][t].to(device)
    tran = char["trans"][t].to(device)
    n_model, n_data = model.num_expression_coeffs, expr.shape[0]
    if n_data >= n_model:
        expr_in = expr[:n_model]
    else:
        expr_in = torch.cat([expr, torch.zeros(n_model - n_data, device=device)])
    out = model(
        global_orient=pose[POSE_IDX["global_orient"]].view(1, 3),
        body_pose=pose[POSE_IDX["body_pose"]].view(1, -1),
        left_hand_pose=pose[POSE_IDX["left_hand_pose"]].view(1, -1),
        right_hand_pose=pose[POSE_IDX["right_hand_pose"]].view(1, -1),
        jaw_pose=pose[POSE_IDX["jaw_pose"]].view(1, 3),
        expression=expr_in.view(1, -1),
        betas=beta.view(1, -1),
        transl=tran.view(1, 3),
    )
    V = out.vertices[0].detach().cpu().numpy()
    J = out.joints[0].detach().cpu().numpy()
    return V, J


# NORMALS (vectorized pre-computation via Open3D)
def precompute_normals(
    verts_frames: List[np.ndarray], faces: np.ndarray
) -> List[np.ndarray]:
    """Compute normals for each frame reusing a single TriangleMesh."""
    V0 = verts_frames[0].astype(np.float64)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V0),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    normals = []
    for V in verts_frames:
        mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
        mesh.compute_vertex_normals()
        normals.append(np.asarray(mesh.vertex_normals).copy())
    return normals


# GROUND
def checker_quads(size=12.0, tile=0.5, y=0.0):
    N = max(1, int(round(size / tile)))
    step = size / N
    half = size * 0.5
    verts, tris, cols = [], [], []
    vid = 0
    z0 = -half
    for iz in range(N):
        x0 = -half
        for ix in range(N):
            v00 = (x0, y, z0)
            v10 = (x0 + step, y, z0)
            v01 = (x0, y, z0 + step)
            v11 = (x0 + step, y, z0 + step)
            verts.extend([v00, v10, v01, v11])
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


# STATE
@dataclass
class State:
    i: int = 0
    playing: bool = True
    speed: float = 1.0
    last: float = field(default_factory=time.perf_counter)
    loop: bool = True


# VIEWER
class Viewer:
    def __init__(self, characters: List[Dict]):
        self.characters = characters
        self.char_idx = 0
        self.show_all = True
        self.show_skeleton = False
        self.shadows_on = False

        # Pre-compute vertices + joints
        self._verts: List[List[np.ndarray]] = []
        self._joints: List[List[np.ndarray]] = []
        self._normals: List[List[np.ndarray]] = []

        for ci, ch in enumerate(self.characters):
            print(f"Precomputing '{ch['name']}' ({ch['T']} frames)")
            vs, js = [], []
            for t in range(ch["T"]):
                V, J = smplx_vertices(ch, t)
                vs.append(V)
                js.append(J)
            # Normals pre-computed outside hot-path
            print(f"  Computing normals")
            faces = get_faces(ch["gender"])
            ns = precompute_normals(vs, faces)
            self._verts.append(vs)
            self._joints.append(js)
            self._normals.append(ns)
            print(f"  Done")

        self.T = min(ch["T"] for ch in self.characters)
        self.fps = self.characters[0]["fps"]
        self.target_dt = 1.0 / max(1.0, float(self.fps))
        self.state = State()
        self._drawn_frame: List[int] = [-1] * len(self.characters)
        self._drawn_active: List[Optional[bool]] = [None] * len(self.characters)

        # Open3D GUI
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("SMPL-X Scene Viewer", 1440, 900)
        em = self.window.theme.font_size
        margin = 0.5 * em

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        # Materials (created once)
        self._mats: List[rendering.MaterialRecord] = []
        for col in CHAR_COLORS:
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            mat.base_color = (*col, 1.0)
            self._mats.append(mat)

        self._mats_dim: List[rendering.MaterialRecord] = []
        for col in CHAR_COLORS:
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            r, g, b = col
            mat.base_color = (r * 0.35, g * 0.35, b * 0.35, 0.85)
            self._mats_dim.append(mat)

        self._mat_lines = rendering.MaterialRecord()
        self._mat_lines.shader = "unlitLine"
        self._mat_lines.line_width = 4.0

        self._mat_lines_active = rendering.MaterialRecord()
        self._mat_lines_active.shader = "unlitLine"
        self._mat_lines_active.line_width = 6.0

        # Ground
        self._y_ground = self._compute_y_ground()
        ground = checker_quads(size=14.0, tile=0.5, y=self._y_ground)
        ground.compute_vertex_normals()
        mat_g = rendering.MaterialRecord()
        mat_g.shader = "defaultLit"
        self.scene_widget.scene.add_geometry("ground", ground, mat_g)

        # Lighting
        self.scene_widget.scene.show_skybox(True)
        self.scene_widget.scene.set_background([0.05, 0.05, 0.17, 1.0])
        self._apply_lighting()

        # pre build meshes
        self._meshes: List[o3d.geometry.TriangleMesh] = []
        for i, ch in enumerate(self.characters):
            faces = get_faces(ch["gender"])
            V0, N0 = self._verts[i][0], self._normals[i][0]
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(V0.astype(np.float64)),
                triangles=o3d.utility.Vector3iVector(faces),
            )
            mesh.vertex_normals = o3d.utility.Vector3dVector(N0.astype(np.float64))
            self._meshes.append(mesh)

        # Camera
        V_all = np.concatenate([self._verts[i][0] for i in range(len(self.characters))])
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(V_all)
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

        # UI PANEL
        self.panel = gui.Vert(0, gui.Margins(margin, margin, margin, margin))

        row_play = gui.Horiz(0.25 * em)
        self.btn_prev = gui.Button("< Prev")
        self.btn_play = gui.Button("Pause")
        self.btn_next = gui.Button("Next >")
        self.btn_prev.set_on_clicked(lambda: self.step(-1))
        self.btn_play.set_on_clicked(self._on_toggle_play)
        self.btn_next.set_on_clicked(lambda: self.step(+1))
        row_play.add_child(self.btn_prev)
        row_play.add_child(self.btn_play)
        row_play.add_child(self.btn_next)
        self.panel.add_child(row_play)

        self.label_frame = gui.Label("Frame: 0 / 0")
        self.panel.add_child(self.label_frame)

        self.panel.add_child(gui.Label("Speed"))
        self.slider_speed = gui.Slider(gui.Slider.DOUBLE)
        self.slider_speed.set_limits(0.1, 3.0)
        self.slider_speed.double_value = 1.0
        self.slider_speed.set_on_value_changed(
            lambda v: setattr(self.state, "speed", float(v))
        )
        self.panel.add_child(self.slider_speed)

        self.cb_show_all = gui.Checkbox("Show all characters")
        self.cb_show_all.checked = self.show_all
        self.cb_show_all.set_on_checked(self._on_show_all)
        self.panel.add_child(self.cb_show_all)

        self.cb_skeleton = gui.Checkbox("Show skeleton")
        self.cb_skeleton.checked = self.show_skeleton
        self.cb_skeleton.set_on_checked(self._on_toggle_skeleton)
        self.panel.add_child(self.cb_skeleton)

        # toggle shadows
        self.cb_shadows = gui.Checkbox("Shadows")
        self.cb_shadows.checked = self.shadows_on
        self.cb_shadows.set_on_checked(self._on_toggle_shadows)
        self.panel.add_child(self.cb_shadows)

        self.panel.add_child(gui.Label("Characters"))
        self.label_active_char = gui.Label(f"Active: {self.characters[0]['name']}")
        self.panel.add_child(self.label_active_char)
        self.char_btns = []
        for i, ch in enumerate(self.characters):
            btn = gui.Button(f"[{i}] {ch['name']} ({ch['gender']})")
            btn.set_on_clicked(lambda i=i: self._set_active_char(i))
            self.panel.add_child(btn)
            self.char_btns.append(btn)

        self.window.add_child(self.panel)

        def on_layout(_):
            r = self.window.content_rect
            panel_w = 280
            self.panel.frame = gui.Rect(r.get_right() - panel_w, r.y, panel_w, r.height)
            self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)

        self.window.set_on_layout(on_layout)
        self._redraw(0, force=True)

    def _apply_lighting(self):
        profile = (
            rendering.Open3DScene.LightingProfile.MED_SHADOWS
            if self.shadows_on
            else rendering.Open3DScene.LightingProfile.NO_SHADOWS
        )
        self.scene_widget.scene.set_lighting(
            profile, np.array([-1.0, -1.5, -0.5], dtype=np.float32)
        )

    def _compute_y_ground(self) -> float:
        y = float("inf")
        for js in self._joints:
            for J in js:
                y = min(y, float(J[:, 1].min()))
        return y

    def mesh_geom_name(self, i):
        return f"mesh_{i}"

    def skel_geom_name(self, i):
        return f"skel_{i}"

    def _redraw(self, t: int, force: bool = False):
        sc = self.scene_widget.scene
        frame_t = min(t, self.T - 1)
        chars_set = set(
            range(len(self.characters)) if self.show_all else [self.char_idx]
        )

        for i in range(len(self.characters)):
            mn = self.mesh_geom_name(i)
            sn = self.skel_geom_name(i)
            is_active = i == self.char_idx
            visible = i in chars_set
            ti = min(frame_t, self.characters[i]["T"] - 1)

            frame_changed = force or (self._drawn_frame[i] != ti)
            active_changed = force or (self._drawn_active[i] != is_active)
            if not visible or not self.show_skeleton:
                if sc.has_geometry(sn):
                    sc.remove_geometry(sn)
            if not visible or self.show_skeleton:
                if sc.has_geometry(mn):
                    sc.remove_geometry(mn)

            if not visible:
                self._drawn_frame[i] = -1
                self._drawn_active[i] = None
                continue

            if not (frame_changed or active_changed):
                continue

            if self.show_skeleton:
                if sc.has_geometry(sn):
                    sc.remove_geometry(sn)
                J = self._joints[i][ti]
                col = [0.6, 0.55, 0.0] if is_active else [0.1, 0.1, 0.1]
                skel = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(J),
                    lines=o3d.utility.Vector2iVector(SKELETON_EDGES_NP),
                )
                skel.colors = o3d.utility.Vector3dVector([col] * len(SKELETON_EDGES))
                mat = self._mat_lines_active if is_active else self._mat_lines
                sc.add_geometry(sn, skel, mat)
            else:
                if sc.has_geometry(mn):
                    sc.remove_geometry(mn)
                V = self._verts[i][ti]
                N = self._normals[i][ti]
                mesh = self._meshes[i]
                mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
                mesh.vertex_normals = o3d.utility.Vector3dVector(N.astype(np.float64))
                mat = (
                    self._mats[i % len(self._mats)]
                    if is_active
                    else self._mats_dim[i % len(self._mats_dim)]
                )
                sc.add_geometry(mn, mesh, mat)

            self._drawn_frame[i] = ti
            self._drawn_active[i] = is_active

        self.label_frame.text = f"Frame: {frame_t} / {self.T - 1}"
        self.window.post_redraw()

    def run(self):
        frame_duration = 1.0 / 60.0
        while self.app.run_one_tick():
            t0 = time.perf_counter()
            self._on_tick()
            elapsed = time.perf_counter() - t0
            remaining = frame_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _on_tick(self):
        now = time.perf_counter()
        dt = now - self.state.last
        target = (
            self.target_dt / max(1e-6, self.state.speed) if self.state.playing else 1e9
        )
        if dt >= target:
            self.state.last = now
            if self.state.playing:
                ni = (
                    (self.state.i + 1) % self.T
                    if self.state.loop
                    else min(self.state.i + 1, self.T - 1)
                )
                self.seek(ni)

    def step(self, di: int):
        i = (
            ((self.state.i + di) % self.T)
            if self.state.loop
            else max(0, min(self.T - 1, self.state.i + di))
        )
        self.seek(i)

    def seek(self, i: int):
        self.state.i = int(i)
        self._redraw(self.state.i)

    def _on_toggle_play(self):
        self.state.playing = not self.state.playing
        self.btn_play.text = "Pause" if self.state.playing else "Play"
        self.state.last = time.perf_counter()

    def _on_toggle_skeleton(self, checked: bool):
        self.show_skeleton = checked
        self._drawn_frame = [-1] * len(self.characters)  # force redraw
        self._drawn_active = [None] * len(self.characters)
        self._redraw(self.state.i, force=True)

    def _on_show_all(self, checked: bool):
        self.show_all = checked
        self._redraw(self.state.i, force=True)

    def _on_toggle_shadows(self, checked: bool):
        self.shadows_on = checked
        self._apply_lighting()
        self.window.post_redraw()

    def _set_active_char(self, i: int):
        self.char_idx = i
        self.fps = self.characters[i]["fps"]
        self.target_dt = 1.0 / max(1.0, float(self.fps))
        self.state.i = min(self.state.i, self.characters[i]["T"] - 1)
        ch = self.characters[i]
        self.label_active_char.text = f"Active: {ch['name']} ({ch['gender']})"
        self.window.title = f"SMPL-X Viewer  {ch['name']} ({ch['gender']})"
        self._drawn_active = [None] * len(self.characters)  # force active color
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
            dx = event.x - self.last_mouse[0]
            dy = event.y - self.last_mouse[1]
            self.last_mouse = (event.x, event.y)
            if event.is_button_down(gui.MouseButton.LEFT):
                self.orbit(dx, dy)
            elif event.is_button_down(gui.MouseButton.RIGHT):
                self.pan(dx, dy)
        elif event.type == gui.MouseEvent.Type.WHEEL:
            self.zoom(event.wheel_dy)
        return gui.SceneWidget.EventCallbackResult.CONSUMED

    def orbit(self, dx, dy):
        sens = 0.01
        self.theta += dx * sens
        self.phi += dy * sens
        self.phi = np.clip(self.phi, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        self.update_camera()

    def zoom(self, dy):
        self.radius *= 1.0 - dy * 0.1
        self.radius = max(0.1, self.radius)
        self.update_camera()

    def pan(self, dx, dy):
        pan_speed = 0.002 * self.radius
        forward = self.center - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        move = -dx * pan_speed * right + dy * pan_speed * up
        self.center += move
        self.eye += move
        self.update_camera()


# MAIN
if __name__ == "__main__":
    chars = load_npz(args.npz, args.start, args.end)
    Viewer(chars).run()
