from __future__ import annotations
import faulthandler

faulthandler.enable()
import argparse, time
from dataclasses import dataclass
import numpy as np
import torch
import open3d as o3d
from open3d.visualization import gui, rendering

ap = argparse.ArgumentParser()
ap.add_argument("--npy", type=str, default="./000000.npy")
ap.add_argument("--npy2", type=str, default=None, help="Second optional anim")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
args = ap.parse_args()

use_cuda = (not args.cpu) and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Device: {device}")


def load_params(npy_path: str, start: int, end: int, device=None):
    data = np.load(npy_path)
    T_all = data.shape[0]
    start = max(0, start)
    end = T_all if end < 0 else min(end, T_all)
    data = data[start:end]
    seq_len = data.shape[0]

    joint_num = 22

    # Root rotation & position
    root_rot_vel_y = data[:, 0]  # rad/s
    root_lin_vel_xz = data[:, 1:3]  # m/s
    root_y = data[:, 3]  # m

    # Integration rotation
    dt = 1 / 30.0  # or args.fps
    root_orient_y = np.cumsum(root_rot_vel_y * dt)

    cos_y = np.cos(root_orient_y)
    sin_y = np.sin(root_orient_y)

    # Apply rot on vel XZ
    vx = root_lin_vel_xz[:, 0] * cos_y - root_lin_vel_xz[:, 1] * sin_y
    vz = root_lin_vel_xz[:, 0] * sin_y + root_lin_vel_xz[:, 1] * cos_y

    root_pos_xz = np.cumsum(np.stack([vx, vz], axis=-1), axis=0)

    # Concat Y
    root_pos = np.concatenate(
        [root_pos_xz[:, 0:1], root_y[:, None], root_pos_xz[:, 1:2]], axis=-1
    )

    # RIC joints
    ric_start = 4
    ric_end = ric_start + (joint_num - 1) * 3
    ric_data = data[:, ric_start:ric_end].reshape(seq_len, joint_num - 1, 3)

    # add root location to other joints
    ric_data_global = ric_data.copy()
    ric_data_global[:, :, 0] += root_pos[:, 0:1]  # X
    ric_data_global[:, :, 2] += root_pos[:, 2:3]
    joints = np.zeros((seq_len, joint_num, 3))
    joints[:, 0, :] = root_pos
    joints[:, 1:, :] = ric_data_global
    return joints.astype(np.float32)


@dataclass
class State:
    i: int = 0
    playing: bool = True
    speed: float = 1.0
    last: float = time.perf_counter()
    loop: bool = True


class Viewer:
    def __init__(self, animations):
        self.show_skeleton = True
        self.animations = animations
        self.anim_index = 0
        self.params_name, self.joints = self.animations[self.anim_index]
        self.T = self.joints.shape[0]
        print(f"Loaded animation {self.params_name} with {self.T} frames.")

        self.state = State()
        self.target_dt = 1.0 / max(1e-6, args.fps)

        # GUI
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("3D Skeleton Viewer", 1280, 800)
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)
        self.mat_lines = rendering.MaterialRecord()
        self.mat_lines.shader = "defaultLit"

        self.traj_geom_names = []
        self.update_traj()

        # Ground
        self.ground = self.checker_quads(size=7.0, tile=2, y=self.y_ground)
        self.ground.compute_vertex_normals()
        mat_ground = rendering.MaterialRecord()
        mat_ground.shader = "defaultLit"
        self.scene.scene.add_geometry("ground", self.ground, mat_ground)

        # Camera setup
        bbox = self.compute_bbox(self.joints[0])
        self.center = bbox.get_center()
        self.eye = self.center + np.array([3.0, 2.0, 3.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.scene.setup_camera(60.0, bbox, self.center)
        self.scene.look_at(self.center, self.eye, self.up)
        offset = self.eye - self.center
        self.radius = np.linalg.norm(offset)
        self.theta = np.arctan2(offset[0], offset[2])
        self.phi = np.arcsin(offset[1] / self.radius)
        self.scene.set_on_mouse(self._on_mouse)

        self.panel = gui.Vert(0, gui.Margins(margin, margin, margin, margin))
        row = gui.Horiz(0.25 * em)
        self.btn_play = gui.Button("Pause")
        self.btn_prev = gui.Button("Prev frame")
        self.btn_next = gui.Button("Next frame")
        row.add_child(self.btn_prev)
        row.add_child(self.btn_play)
        row.add_child(self.btn_next)
        self.panel.add_child(row)

        self.checkbox_skeleton = gui.Checkbox("Show skeleton")
        self.checkbox_skeleton.checked = self.show_skeleton
        self.checkbox_skeleton.set_on_checked(self.on_toggle_skeleton)
        self.panel.add_child(self.checkbox_skeleton)

        self.label_file = gui.Label(f"Fichier: {self.params_name}")
        self.panel.add_child(self.label_file)

        if len(self.animations) > 1:
            self.btn_switch = gui.Button("Switch animation")
            self.btn_switch.set_on_clicked(self.on_switch_animation)
            self.panel.add_child(self.btn_switch)

        self.window.add_child(self.panel)

        def on_layout(_):
            r = self.window.content_rect
            panel_w = 300
            self.panel.frame = gui.Rect(r.get_right() - panel_w, r.y, panel_w, r.height)
            self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)

        self.window.set_on_layout(on_layout)
        self.btn_play.set_on_clicked(self.on_toggle_play)
        self.btn_prev.set_on_clicked(lambda: self.step(-1))
        self.btn_next.set_on_clicked(lambda: self.step(+1))

        self.slider_speed = gui.Slider(gui.Slider.DOUBLE)
        self.slider_speed.set_limits(0.1, 2.0)
        self.slider_speed.double_value = 1.0
        self.slider_speed.set_on_value_changed(self.on_speed_changed)
        self.panel.add_child(gui.Label("Animation speed"))
        self.panel.add_child(self.slider_speed)

        self._last_mouse = None
        self.update_geometry(self.joints[0])

    def create_skeleton(self, joints: np.ndarray) -> o3d.geometry.TriangleMesh:
        edges = [
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

        skeleton_mesh = o3d.geometry.TriangleMesh()
        radius = 0.02

        for a, b in edges:
            start = joints[a]
            end = joints[b]
            vec = end - start
            length = np.linalg.norm(vec)
            if length < 1e-6:
                continue
            cyl = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius, height=length, resolution=10, split=1
            )
            cyl.compute_vertex_normals()

            z_axis = np.array([0, 0, 1])
            vec_norm = vec / length
            R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0])  # identity
            if not np.allclose(vec_norm, z_axis):
                # rotate around axis
                axis = np.cross(z_axis, vec_norm)
                axis_len = np.linalg.norm(axis)
                if axis_len > 1e-6:
                    axis = axis / axis_len
                    angle = np.arccos(np.dot(z_axis, vec_norm))
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                elif np.dot(z_axis, vec_norm) < 0:  # opposed vectors
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                        np.pi * np.array([1, 0, 0])
                    )

            cyl.rotate(R, center=np.zeros(3))
            cyl.translate((start + end) / 2)
            skeleton_mesh += cyl

        skeleton_mesh.paint_uniform_color([1, 0.25, 0])
        return skeleton_mesh

    def update_traj(self):
        for name in self.traj_geom_names:
            self.scene.scene.remove_geometry(name)
        self.traj_geom_names.clear()
        self.traj = []
        y_ground = float("inf")
        for t in range(self.T):
            J = self.joints[t]
            y_ground = min(y_ground, float(J[:, 1].min()))
            self.traj.append([J[0, 0], J[0, 2]])
        self.y_ground = y_ground
        print("y_ground =", y_ground)
        for idx, (x, z) in enumerate(self.traj):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.translate([x, self.y_ground, z])
            sphere.paint_uniform_color([0.8, 0.8, 0.8])
            mat = rendering.MaterialRecord()
            geom_name = f"sphere_{idx}"
            self.scene.scene.add_geometry(geom_name, sphere, mat)
            self.traj_geom_names.append(geom_name)

    def checker_quads(self, size=15.0, tile=1, y=0.0):
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
                if (ix + iz) % 2 == 0:
                    cols.extend([(0.1, 0.5, 0.9)] * 4)
                else:
                    cols.extend([(0.2, 0.7, 1.0)] * 4)
                vid += 4
                x0 += step
            z0 += step
        m = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(verts, np.float64)),
            triangles=o3d.utility.Vector3iVector(np.asarray(tris, np.int32)),
        )
        m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cols, np.float64))
        return m

    def compute_bbox(self, joints):
        bbox = o3d.geometry.AxisAlignedBoundingBox()
        bbox.min_bound = joints.min(axis=0)
        bbox.max_bound = joints.max(axis=0)
        return bbox

    def update_geometry(self, J: np.ndarray):
        if self.scene.scene.has_geometry("skeleton"):
            self.scene.scene.remove_geometry("skeleton")
        if self.show_skeleton:
            skeleton = self.create_skeleton(J)
            self.scene.scene.add_geometry("skeleton", skeleton, self.mat_lines)
        self.window.post_redraw()

    def run(self):
        while self.app.run_one_tick():
            self.on_tick()
            time.sleep(1.0 / 60.0)

    def on_toggle_play(self):
        self.state.playing = not self.state.playing
        self.btn_play.text = "⏸ Pause" if self.state.playing else "▶ Play"
        self.state.last = time.perf_counter()

    def on_tick(self):
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
            (self.state.i + di) % self.T
            if self.state.loop
            else max(0, min(self.T - 1, self.state.i + di))
        )
        self.seek(i)

    def seek(self, i: int):
        self.state.i = int(i)
        self.update_geometry(self.joints[i])

    def on_toggle_skeleton(self, checked: bool):
        self.show_skeleton = checked
        self.update_geometry(self.joints[self.state.i])

    def on_speed_changed(self, value):
        self.state.speed = float(value)

    def on_switch_animation(self):
        self.anim_index = (self.anim_index + 1) % len(self.animations)
        self.params_name, self.joints = self.animations[self.anim_index]
        self.T = self.joints.shape[0]
        self.state.i = 0
        self.update_traj()
        self.label_file.text = f"Fichier: {self.params_name}"
        self.update_geometry(self.joints[0])

    def _update_camera(self):
        x = self.radius * np.cos(self.phi) * np.sin(self.theta)
        y = self.radius * np.sin(self.phi)
        z = self.radius * np.cos(self.phi) * np.cos(self.theta)
        self.eye = np.array([x, y, z]) + self.center
        self.scene.look_at(self.center, self.eye, self.up)
        self.scene.force_redraw()

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self._last_mouse = (event.x, event.y)
        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            self._last_mouse = None
        elif event.type == gui.MouseEvent.Type.DRAG and self._last_mouse is not None:
            dx = event.x - self._last_mouse[0]
            dy = event.y - self._last_mouse[1]
            self._last_mouse = (event.x, event.y)
            if event.is_button_down(gui.MouseButton.LEFT):
                self._orbit(dx, dy)
            elif event.is_button_down(gui.MouseButton.RIGHT):
                self._pan(dx, dy)
        elif event.type == gui.MouseEvent.Type.WHEEL:
            self._zoom(event.wheel_dy)
        return gui.SceneWidget.EventCallbackResult.CONSUMED

    def _orbit(self, dx, dy):
        sens = 0.01
        self.theta += dx * sens
        self.phi += dy * sens
        self.phi = np.clip(self.phi, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        self._update_camera()

    def _zoom(self, dy):
        zoom_speed = 0.1
        self.radius *= 1.0 - dy * zoom_speed
        self.radius = max(0.1, self.radius)
        self._update_camera()

    def _pan(self, dx, dy):
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
        self._update_camera()


if __name__ == "__main__":
    animations = []
    animations.append((args.npy, load_params(args.npy, args.start, args.end, device)))
    if args.npy2 is not None:
        animations.append(
            (args.npy2, load_params(args.npy2, args.start, args.end, device))
        )
    Viewer(animations).run()
