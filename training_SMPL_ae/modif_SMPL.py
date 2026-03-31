from __future__ import annotations
import faulthandler

faulthandler.enable()
import argparse, time
from dataclasses import dataclass
import numpy as np
import torch
import open3d as o3d
from open3d.visualization import gui, rendering
import smplx
from smplx import SMPLX
from utils.rotation_conversions import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
)

ap = argparse.ArgumentParser()
ap.add_argument("--npy", type=str, default="./000000.npy")
ap.add_argument("--npy2", type=str, default=None)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
args = ap.parse_args()

device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
print(f"Device: {device}")


# root_rot_vel_y              [0:1]	  Vélocité angulaire Y
# root_lin_vel_xz             [1:3]     Vélocité position XZ de la root
# root_y_pos 			     [3]       Position Y de la root
# joints_pos                  [4:67]    Position 3D des jointures (21 joints hors root)
# joints_rot                  [67:193]  Rotations 6D des jointures (6D, 21 joints hors root)
# joints_vel                  [193:259] Vélocité des jointures
# Foot Contact foot_contact   [259:263] Contacts au sol des pieds (On les a partiellement)


def load_params(npy_path: str, start: int, end: int, device=None):
    data = np.load(npy_path)
    T = data.shape[0]
    start, end = max(0, start), T if end < 0 else min(end, T)
    data = data[start:end]
    seq_len = data.shape[0]

    root_rot_vel_y, root_lin_vel_xz, root_y = data[:, 0], data[:, 1:3], data[:, 3]
    dt = 1 / 30.0
    root_orient_y = np.cumsum(root_rot_vel_y * dt)
    cos_y, sin_y = np.cos(root_orient_y), np.sin(root_orient_y)

    vx = root_lin_vel_xz[:, 0] * cos_y - root_lin_vel_xz[:, 1] * sin_y
    vz = root_lin_vel_xz[:, 0] * sin_y + root_lin_vel_xz[:, 1] * cos_y
    root_pos_xz = np.cumsum(np.stack([vx, vz], axis=-1), axis=0)
    root_pos = np.concatenate(
        [root_pos_xz[:, 0:1], root_y[:, None], root_pos_xz[:, 1:2]], axis=-1
    )

    ric_data = data[:, 4:67].reshape(seq_len, 21, 3)
    ric_data[:, :, 0] += root_pos[:, 0:1]
    ric_data[:, :, 2] += root_pos[:, 2:3]

    joints = np.zeros((seq_len, 22, 3))
    joints[:, 0, :], joints[:, 1:, :] = root_pos, ric_data
    return joints.astype(np.float32)


# def load_params_using_smplx_and_opti(
#    npy_path: str,
#    start: int = 0,
#    end: int = -1,
#    device=None,
#    smplx_model_path: str = "body_models",
#    num_iters: int = 50,
# ):
#    device = torch.device(
#        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
#    )
#    data = np.load(npy_path)
#    T = data.shape[0]
#    start = max(0, start)
#    end = T if end < 0 else min(end, T)
#    data = data[start:end]
#    seq_len = data.shape[0]
#    smplx_model = smplx.create(
#        model_path=smplx_model_path,
#        model_type="smplx",
#        gender="neutral",
#        use_pca=False,
#        num_betas=10,
#        batch_size=1,
#    ).to(device)
#
#    dt = 1 / 30.0
#    root_rot_vel_y = data[:, 0]
#    root_lin_vel_xz = data[:, 1:3]
#    root_y = data[:, 3]
#
#    root_orient_y = np.cumsum(root_rot_vel_y * dt)
#    cos_y, sin_y = np.cos(root_orient_y), np.sin(root_orient_y)
#
#    vx = root_lin_vel_xz[:, 0] * cos_y - root_lin_vel_xz[:, 1] * sin_y
#    vz = root_lin_vel_xz[:, 0] * sin_y + root_lin_vel_xz[:, 1] * cos_y
#    root_pos_xz = np.cumsum(np.stack([vx, vz], axis=-1), axis=0)
#
#    root_pos = np.stack([root_pos_xz[:, 0], root_y, root_pos_xz[:, 1]], axis=-1)
#
#    all_joints = []
#
#    for t in range(seq_len):
#        joints_body = torch.from_numpy(data[t, 4:67].reshape(21, 3)).float().to(device)
#        root_position = torch.from_numpy(root_pos[t]).float().unsqueeze(0).to(device)
#        joints_target = torch.cat([root_position, joints_body], dim=0)  # (22,3)
#        body_pose = torch.zeros(1, 21 * 3, device=device, requires_grad=True)
#        global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
#        transl = root_position.clone().to(device)
#        transl.requires_grad = True
#
#        optimizer = torch.optim.LBFGS(
#            [body_pose, global_orient, transl], lr=1.0, max_iter=num_iters
#        )
#        def closure():
#            optimizer.zero_grad()
#            output = smplx_model(
#                body_pose=body_pose,
#                global_orient=global_orient,
#                transl=transl,
#                return_verts=False,
#            )
#            joints_smplx = output.joints[:, :22]
#            loss = torch.mean((joints_smplx - joints_target) ** 2)
#            loss.backward()
#            return loss
#        optimizer.step(closure)
#        with torch.no_grad():
#            output = smplx_model(
#                body_pose=body_pose,
#                global_orient=global_orient,
#                transl=transl,
#                return_verts=False,
#            )
#            joints_frame = output.joints[:, :22, :].cpu().numpy().reshape(22, 3)
#            all_joints.append(joints_frame)
#    all_joints = np.stack(all_joints, axis=0)  # (T, 22, 3)
#    return all_joints.astype(np.float32)


@dataclass
class State:
    i: int = 0
    playing: bool = False
    speed: float = 1.0
    last: float = time.perf_counter()
    loop: bool = True
    edit_mode: bool = True
    selected_joint: int = -1
    dragging: bool = False


class Viewer:
    def __init__(self, animations):
        self.show_skeleton = True
        self.show_all_frames = False
        self.animations = animations
        self.anim_index = 0
        self.params_name, self.joints = self.animations[0]
        self.joints = self.joints.copy()
        self.T = self.joints.shape[0]
        self.state = State()
        self.target_dt = 1.0 / max(1e-6, args.fps)

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("3D Skeleton Editor", 1280, 800)
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.7, 0.7, 0.8, 1.0])
        self.window.add_child(self.scene)
        self.mat_lines = rendering.MaterialRecord()
        self.mat_lines.shader = "defaultLit"

        self.traj_geom_names = []
        self.update_traj()

        self.ground = self.checker_quads(14.0, 1, self.y_ground)
        self.ground.compute_vertex_normals()
        mat_ground = rendering.MaterialRecord()
        mat_ground.shader = "defaultLit"
        self.scene.scene.add_geometry("ground", self.ground, mat_ground)

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
        self.modified_flags = np.zeros(
            (self.T, 22), dtype=bool
        )  # Flags pour modifications
        self.pc_all_frames = None
        self.lines_all_frames = None

        # GUI
        self.panel = gui.Vert(0, gui.Margins(margin, margin, margin, margin))

        self.checkbox_edit = gui.Checkbox("Edit Mode")
        self.checkbox_edit.checked = True
        self.checkbox_edit.set_on_checked(
            lambda c: setattr(self.state, "edit_mode", c) or self.on_toggle_edit()
        )
        self.panel.add_child(self.checkbox_edit)

        self.frame_nb = gui.Label(f"Frame {self.anim_index}")
        self.panel.add_child(self.frame_nb)

        row = gui.Horiz(0.25 * em)
        self.btn_play = gui.Button("Play")
        self.btn_prev = gui.Button("Prev frame")
        self.btn_next = gui.Button("Next frame")
        row.add_child(self.btn_prev)
        row.add_child(self.btn_play)
        row.add_child(self.btn_next)
        self.panel.add_child(row)

        self.checkbox_skeleton = gui.Checkbox("Show skeleton")
        self.checkbox_skeleton.checked = True
        self.checkbox_skeleton.set_on_checked(
            lambda c: setattr(self, "show_skeleton", c)
            or self.update_geometry(self.joints[self.state.i])
        )
        self.panel.add_child(self.checkbox_skeleton)

        self.checkbox_all_frames = gui.Checkbox("Show all frames")
        self.checkbox_all_frames.checked = False
        self.checkbox_all_frames.set_on_checked(
            lambda c: setattr(self, "show_all_frames", c)
            or self.init_all_frames_geometry()
        )
        self.panel.add_child(self.checkbox_all_frames)

        self.label_info = gui.Label("Click on a joint to select it")
        self.panel.add_child(self.label_info)
        self.label_file = gui.Label(f"Fichier: {self.params_name}")
        self.panel.add_child(self.label_file)

        self.btn_save = gui.Button("Save edited animation")
        self.btn_save.set_on_clicked(self.on_save)
        self.panel.add_child(self.btn_save)

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
        self.slider_speed.set_on_value_changed(
            lambda v: setattr(self.state, "speed", float(v))
        )
        self.panel.add_child(gui.Label("Animation speed"))
        self.panel.add_child(self.slider_speed)

        self._last_mouse = None

        # Init optimized all frames geometry
        self.pc_all_frames = None
        self.lines_all_frames = None

        self.update_geometry(self.joints[0])

    def update_all_frames_pc_colors(self):
        """Met à jour les positions et couleurs de toutes les frames dans show_all_frames."""
        if self.pc_all_frames is None:
            return

        points = np.asarray(self.pc_all_frames.points)
        colors = np.asarray(self.pc_all_frames.colors)

        for t in range(self.T):
            for i in range(22):
                idx = t * 22 + i
                points[idx] = self.joints[t, i]
                if t == self.state.i and i == self.state.selected_joint:
                    colors[idx] = [1, 0, 0.5]  # jaune sélection
                elif t == self.state.i:
                    colors[idx] = (
                        [0, 1.0, 0.7]
                        if not self.modified_flags[t, i]
                        else [1.0, 0.0, 0.3]
                    )
                else:
                    colors[idx] = (
                        [0.5, 0.5, 0.5]
                        if not self.modified_flags[t, i]
                        else [1.0, 0.0, 0.0]
                    )

        # Remove old geometry
        self.scene.scene.remove_geometry("all_joints_pc")
        self.scene.scene.remove_geometry("all_trails")

        # Recreate point cloud
        self.pc_all_frames.points = o3d.utility.Vector3dVector(points)
        self.pc_all_frames.colors = o3d.utility.Vector3dVector(colors)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 5.0
        self.scene.scene.add_geometry("all_joints_pc", self.pc_all_frames, mat)

        # Recreate lines
        if self.lines_all_frames:
            line_points = np.asarray(self.lines_all_frames.points)
            line_points[:] = points
            self.lines_all_frames.points = o3d.utility.Vector3dVector(line_points)
            mat_line = rendering.MaterialRecord()
            mat_line.shader = "unlitLine"
            mat_line.line_width = 1.0
            self.scene.scene.add_geometry("all_trails", self.lines_all_frames, mat_line)

    def init_all_frames_geometry(self):
        # Remove old geometry if present
        if self.pc_all_frames:
            self.scene.scene.remove_geometry("all_joints_pc")
        if self.lines_all_frames:
            self.scene.scene.remove_geometry("all_trails")

        # Points cloud for all joints
        points = self.joints.reshape(-1, 3)
        self.pc_all_frames = o3d.geometry.PointCloud()
        self.pc_all_frames.points = o3d.utility.Vector3dVector(points)
        colors = np.full_like(points, 0.5)  # gris par défaut
        self.pc_all_frames.colors = o3d.utility.Vector3dVector(colors)
        self.scene.scene.add_geometry(
            "all_joints_pc", self.pc_all_frames, rendering.MaterialRecord()
        )

        # Lines between consecutive frames
        lines = []
        line_colors = []
        for t in range(self.T - 1):
            for i in range(22):
                idx0, idx1 = t * 22 + i, (t + 1) * 22 + i
                lines.append([idx0, idx1])
                line_colors.append([0.3, 0.3, 0.3])
        self.lines_all_frames = o3d.geometry.LineSet()
        self.lines_all_frames.points = o3d.utility.Vector3dVector(points)
        self.lines_all_frames.lines = o3d.utility.Vector2iVector(lines)
        self.lines_all_frames.colors = o3d.utility.Vector3dVector(line_colors)
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 1.0
        self.scene.scene.add_geometry("all_trails", self.lines_all_frames, mat)

        self.update_geometry(self.joints[self.state.i])

    #  Skeleton
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
        skeleton = o3d.geometry.TriangleMesh()
        for a, b in edges:
            vec = joints[b] - joints[a]
            length = np.linalg.norm(vec)
            if length < 1e-6:
                continue
            cyl = o3d.geometry.TriangleMesh.create_cylinder(0.02, length, 10, 1)
            cyl.compute_vertex_normals()
            vec_norm = vec / length
            z_axis = np.array([0, 0, 1])
            if not np.allclose(vec_norm, z_axis):
                axis = np.cross(z_axis, vec_norm)
                axis_len = np.linalg.norm(axis)
                if axis_len > 1e-6:
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                        axis / axis_len * np.arccos(np.dot(z_axis, vec_norm))
                    )
                elif np.dot(z_axis, vec_norm) < 0:
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                        np.pi * np.array([1, 0, 0])
                    )
                else:
                    R = np.eye(3)
                cyl.rotate(R, center=np.zeros(3))
            cyl.translate((joints[a] + joints[b]) / 2)
            skeleton += cyl
        skeleton.paint_uniform_color([0.1, 0.4, 0.3])
        return skeleton

    #  Update Geometry

    def update_geometry(self, J: np.ndarray):
        # Skeleton
        if self.scene.scene.has_geometry("skeleton"):
            self.scene.scene.remove_geometry("skeleton")
        if self.show_skeleton:
            self.scene.scene.add_geometry(
                "skeleton", self.create_skeleton(J), self.mat_lines
            )

        # Current frame spheres
        for i in range(22):
            name = f"joint_{i}"
            if self.scene.scene.has_geometry(name):
                self.scene.scene.remove_geometry(name)
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.04)
            sphere.translate(J[i])
            color = (
                [1, 0, 0.5]
                if i == self.state.selected_joint
                else (
                    [0.0, 1.0, 0.7]
                    if not self.modified_flags[self.state.i, i]
                    else [1.0, 0.0, 0.3]
                )
            )
            sphere.paint_uniform_color(color)
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"

            sphere.compute_vertex_normals()
            self.scene.scene.add_geometry(name, sphere, mat)

        #  Show All Frames
        if self.show_all_frames and self.pc_all_frames:
            points = np.asarray(self.pc_all_frames.points)
            colors = np.asarray(self.pc_all_frames.colors)
            for t in range(self.T):
                for i in range(22):
                    idx = t * 22 + i
                    points[idx] = self.joints[t, i]  # Update position
                    # Color if modified and selected frame
                    if t == self.state.i and i == self.state.selected_joint:
                        colors[idx] = [1, 0, 0.5]  # yellow sélection
                    elif t == self.state.i:
                        colors[idx] = (
                            [0, 1.0, 0.7]  # green/blue if not modified
                            if not self.modified_flags[t, i]
                            else [1.0, 0.0, 0.3]  # magenta if modified
                        )
                    else:
                        colors[idx] = (
                            [0.5, 0.5, 0.5]  # grey if modified
                            if not self.modified_flags[t, i]
                            else [1.0, 0.0, 0.0]  # red if modifiesd
                        )
            self.pc_all_frames.points = o3d.utility.Vector3dVector(points)
            self.pc_all_frames.colors = o3d.utility.Vector3dVector(colors)

            # Update line points (trails)
            line_points = np.asarray(self.lines_all_frames.points)
            line_points[:] = points
            self.lines_all_frames.points = o3d.utility.Vector3dVector(line_points)
        else:
            if self.scene.scene.has_geometry(
                "all_joints_pc"
            ) and self.scene.scene.has_geometry("all_trails"):
                self.scene.scene.remove_geometry("all_joints_pc")
                self.scene.scene.remove_geometry("all_trails")
        self.window.post_redraw()

    # --- Trajectory ---
    def update_traj(self):
        for name in self.traj_geom_names:
            self.scene.scene.remove_geometry(name)
        self.traj_geom_names.clear()
        y_ground = float("inf")
        for t in range(self.T):
            y_ground = min(y_ground, float(self.joints[t, :, 1].min()))
        self.y_ground = y_ground
        for idx in range(self.T):
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.02)
            sphere.translate(
                [self.joints[idx, 0, 0], self.y_ground, self.joints[idx, 0, 2]]
            )
            sphere.paint_uniform_color([0.8, 0.8, 0.8])
            name = f"sphere_{idx}"
            self.scene.scene.add_geometry(name, sphere, rendering.MaterialRecord())
            self.traj_geom_names.append(name)

    # --- Checker Ground ---
    def checker_quads(self, size=15.0, tile=1, y=0.0):
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
                cols.extend(
                    [(0.1, 0.5, 0.9) if (ix + iz) % 2 == 0 else (0.2, 0.7, 1.0)] * 4
                )
                vid += 4
                x0 += step
            z0 += step
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(verts, np.float64)),
            o3d.utility.Vector3iVector(np.asarray(tris, np.int32)),
        )
        m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cols, np.float64))
        return m

    # --- Bounding box ---
    def compute_bbox(self, joints):
        bbox = o3d.geometry.AxisAlignedBoundingBox()
        bbox.min_bound, bbox.max_bound = joints.min(axis=0), joints.max(axis=0)
        return bbox

    def find_closest_joint(self, x: int, y: int) -> int:
        w, h = self.scene.frame.width, self.scene.frame.height
        ndc_x, ndc_y = (2.0 * x / w) - 1.0, 1.0 - (2.0 * y / h)

        J = self.joints[self.state.i]
        view_dir = (self.center - self.eye) / np.linalg.norm(self.center - self.eye)
        right = np.cross(view_dir, self.up)
        right /= np.linalg.norm(right)
        up_cam = np.cross(right, view_dir)

        min_dist, closest = float("inf"), -1
        for i in range(22):
            offset = J[i] - self.eye
            dist_3d = np.linalg.norm(offset)
            screen_x = np.dot(offset, right) / dist_3d
            screen_y = np.dot(offset, up_cam) / dist_3d
            dist_2d = np.sqrt(
                (screen_x - ndc_x * 0.5) ** 2 + (screen_y - ndc_y * 0.5) ** 2
            )
            if dist_2d < min_dist and dist_2d < 0.15:
                min_dist, closest = dist_2d, i
        return closest

    def screen_to_world_plane(self, x: int, y: int, plane_point: np.ndarray):
        w, h = self.scene.frame.width, self.scene.frame.height
        ndc_x, ndc_y = (2.0 * x / w) - 1.0, 1.0 - (2.0 * y / h)

        view_dir = (self.center - self.eye) / np.linalg.norm(self.center - self.eye)
        right = np.cross(view_dir, self.up) / np.linalg.norm(
            np.cross(view_dir, self.up)
        )
        up_cam = np.cross(right, view_dir) / np.linalg.norm(np.cross(right, view_dir))

        dist = np.dot(plane_point - self.eye, view_dir)
        fov_rad = np.radians(60.0)
        half_h = dist * np.tan(fov_rad / 2.0)
        half_w = half_h * (w / h)

        return (
            self.eye
            + view_dir * dist
            + right * (ndc_x * half_w)
            + up_cam * (ndc_y * half_h)
        )

    def move_joint_to_screen_pos(self, x: int, y: int):
        if self.state.selected_joint >= 0:
            J = self.joints[self.state.i]
            new_pos = self.screen_to_world_plane(x, y, J[self.state.selected_joint])
            if not np.allclose(J[self.state.selected_joint], new_pos):
                self.joints[self.state.i, self.state.selected_joint] = new_pos
                self.modified_flags[self.state.i, self.state.selected_joint] = True
            # Update current frame spheres
            self.update_geometry(J)
            # Update all frames pointcloud
        if self.show_all_frames:
            self.update_all_frames_pc_colors()

    def run(self):
        while self.app.run_one_tick():
            self.on_tick()
            time.sleep(1.0 / 60.0)

    def on_toggle_play(self):
        self.state.playing = not self.state.playing
        self.btn_play.text = "Pause" if self.state.playing else "Play"
        self.state.last = time.perf_counter()

    def on_tick(self):
        now = time.perf_counter()
        if now - self.state.last >= (
            self.target_dt / max(1e-6, self.state.speed) if self.state.playing else 1e9
        ):
            self.state.last = now
            if self.state.playing:
                self.seek(
                    (self.state.i + 1) % self.T
                    if self.state.loop
                    else min(self.state.i + 1, self.T - 1)
                )

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
        self.frame_nb.text = f"Frame {i}/{len(self.animations[self.anim_index][1])}"

    def on_toggle_edit(self):
        if not self.state.edit_mode:
            self.state.selected_joint = -1
            self.state.dragging = False
            self.update_geometry(self.joints[self.state.i])
        self.label_info.text = f"Edit Mode: {'ON - Click joints to select' if self.state.edit_mode else 'OFF'}"

    def on_switch_animation(self):
        self.anim_index = (self.anim_index + 1) % len(self.animations)

        self.params_name, self.joints = self.animations[self.anim_index]
        self.joints = self.joints.copy()
        self.T = self.joints.shape[0]
        self.state.i = 0
        self.state.selected_joint = -1
        self.modified_flags = np.zeros((self.T, 22), dtype=bool)
        if self.show_all_frames:
            self.show_all_frames = False
            if self.scene.scene.has_geometry("all_joints_pc"):
                self.scene.scene.remove_geometry("all_joints_pc")
            if self.scene.scene.has_geometry("all_trails"):
                self.scene.scene.remove_geometry("all_trails")
            self.pc_all_frames = None
            self.lines_all_frames = None
            self.show_all_frames = True
            self.init_all_frames_geometry()

        self.update_traj()
        self.label_file.text = f"Fichier: {self.params_name}"
        self.frame_nb.text = f"Frame 0/{len(self.animations[self.anim_index][1])}"
        self.update_geometry(self.joints[0])

    def on_save(self):
        path = self.params_name.replace(".npy", "_edited.npy")
        np.save(path, self.joints)
        self.label_info.text = f"Saved to {path}"

    def _update_camera(self):
        x = self.radius * np.cos(self.phi) * np.sin(self.theta)
        y = self.radius * np.sin(self.phi)
        z = self.radius * np.cos(self.phi) * np.cos(self.theta)
        self.eye = np.array([x, y, z]) + self.center
        self.scene.look_at(self.center, self.eye, self.up)
        self.scene.force_redraw()

    def _on_mouse(self, event):
        if self.state.edit_mode:
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(
                gui.MouseButton.LEFT
            ):
                joint_idx = self.find_closest_joint(event.x, event.y)
                if joint_idx >= 0:
                    self.state.selected_joint, self.state.dragging = joint_idx, True
                    self.label_info.text = f"Selected joint {joint_idx}"
                    self.move_joint_to_screen_pos(event.x, event.y)
                    return gui.SceneWidget.EventCallbackResult.CONSUMED
                else:
                    self.state.selected_joint = -1
                    self.update_geometry(self.joints[self.state.i])
            elif event.type == gui.MouseEvent.Type.BUTTON_UP and event.is_button_down(
                gui.MouseButton.LEFT
            ):
                self.state.dragging = False
                return gui.SceneWidget.EventCallbackResult.CONSUMED
            elif (
                event.type == gui.MouseEvent.Type.DRAG
                and self.state.dragging
                and self.state.selected_joint >= 0
                and event.is_button_down(gui.MouseButton.LEFT)
            ):
                self.move_joint_to_screen_pos(event.x, event.y)
                return gui.SceneWidget.EventCallbackResult.CONSUMED
            elif (
                event.type == gui.MouseEvent.Type.MOVE
                and self.state.dragging
                and self.state.selected_joint >= 0
            ):
                self.move_joint_to_screen_pos(event.x, event.y)
                return gui.SceneWidget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self._last_mouse = (event.x, event.y)
        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            self._last_mouse = None
        elif event.type == gui.MouseEvent.Type.DRAG and self._last_mouse:
            dx, dy = event.x - self._last_mouse[0], event.y - self._last_mouse[1]
            self._last_mouse = (event.x, event.y)
            if event.is_button_down(gui.MouseButton.RIGHT):
                self.theta += dx * 0.01
                self.phi = np.clip(
                    self.phi + dy * 0.01, -np.pi / 2 + 0.01, np.pi / 2 - 0.01
                )
                self._update_camera()
            elif event.is_button_down(gui.MouseButton.MIDDLE):
                forward = (self.center - self.eye) / np.linalg.norm(
                    self.center - self.eye
                )
                right = np.cross(forward, self.up) / np.linalg.norm(
                    np.cross(forward, self.up)
                )
                up = np.cross(right, forward) / np.linalg.norm(np.cross(right, forward))
                move = -dx * 0.002 * self.radius * right + dy * 0.002 * self.radius * up
                self.center += move
                self.eye += move
                self._update_camera()
        elif event.type == gui.MouseEvent.Type.WHEEL:
            self.radius = max(0.1, self.radius * (1.0 - event.wheel_dy * 0.1))
            self._update_camera()
        return gui.SceneWidget.EventCallbackResult.CONSUMED


if __name__ == "__main__":
    animations = [(args.npy, load_params(args.npy, args.start, args.end, device))]
    if args.npy2:
        animations.append(
            (args.npy2, load_params(args.npy2, args.start, args.end, device))
        )
    Viewer(animations).run()
