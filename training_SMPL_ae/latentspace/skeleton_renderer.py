"""skeleton_renderer.py - 3D skeleton visualization with Open3D"""

import numpy as np
import open3d as o3d


class SkeletonRenderer:
    """Open3D-based skeleton renderer with cylindrical bones"""

    EDGES = [
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

    @staticmethod
    def motion_to_joints(recon_np):
        """Convert 263-dim motion representation to 3D joints"""
        seq_len = recon_np.shape[0]
        joint_num = 22
        dt = 1 / 30.0

        root_rot_vel_y = recon_np[:, 0]
        root_lin_vel_xz = recon_np[:, 1:3]
        root_y = recon_np[:, 3]

        root_orient_y = np.cumsum(root_rot_vel_y * dt)
        cos_y = np.cos(root_orient_y)
        sin_y = np.sin(root_orient_y)

        vx = root_lin_vel_xz[:, 0] * cos_y - root_lin_vel_xz[:, 1] * sin_y
        vz = root_lin_vel_xz[:, 0] * sin_y + root_lin_vel_xz[:, 1] * cos_y
        root_pos_xz = np.cumsum(np.stack([vx, vz], axis=-1), axis=0)
        root_pos = np.concatenate(
            [root_pos_xz[:, 0:1], root_y[:, None], root_pos_xz[:, 1:2]], axis=-1
        )

        ric_start = 4
        ric_end = ric_start + (joint_num - 1) * 3
        ric_local = recon_np[:, ric_start:ric_end].reshape(seq_len, joint_num - 1, 3)

        ric_global = ric_local.copy()
        ric_global[:, :, 0] += root_pos[:, 0:1]
        ric_global[:, :, 2] += root_pos[:, 2:3]

        joints = np.zeros((seq_len, joint_num, 3))
        joints[:, 0, :] = root_pos
        joints[:, 1:, :] = ric_global

        return joints

    @staticmethod
    def create_skeleton_mesh(
        joints: np.ndarray, radius: float = 0.02, color: np.array = [1.0, 0.25, 0.0]
    ) -> o3d.geometry.TriangleMesh:
        """Create cylindrical skeleton mesh from joints"""
        skeleton_mesh = o3d.geometry.TriangleMesh()

        for a, b in SkeletonRenderer.EDGES:
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
            R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0])

            if not np.allclose(vec_norm, z_axis):
                axis = np.cross(z_axis, vec_norm)
                axis_len = np.linalg.norm(axis)
                if axis_len > 1e-6:
                    axis = axis / axis_len
                    angle = np.arccos(np.dot(z_axis, vec_norm))
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                elif np.dot(z_axis, vec_norm) < 0:
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                        np.pi * np.array([1, 0, 0])
                    )

            cyl.rotate(R, center=np.zeros(3))
            cyl.translate((start + end) / 2)
            skeleton_mesh += cyl

        skeleton_mesh.paint_uniform_color(color)
        return skeleton_mesh

    @staticmethod
    def create_ground_plane(size: float = 7.0, tile: float = 2.0, y_ground: float = 0):
        """Create checkered ground plane"""
        N = max(1, int(round(size / tile)))
        step = size / N
        half = size * 0.5
        verts, tris, cols = [], [], []
        vid = 0

        z0 = -half
        for iz in range(N):
            x0 = -half
            for ix in range(N):
                v00 = (x0, y_ground, z0)
                v10 = (x0 + step, y_ground, z0)
                v01 = (x0, y_ground, z0 + step)
                v11 = (x0 + step, y_ground, z0 + step)
                verts.extend([v00, v10, v01, v11])
                tris.extend([(vid, vid + 2, vid + 1), (vid + 1, vid + 2, vid + 3)])

                color = (
                    [(0.1, 0.5, 0.9)] * 4
                    if (ix + iz) % 2 == 0
                    else [(0.2, 0.7, 1.0)] * 4
                )
                cols.extend(color)
                vid += 4
                x0 += step
            z0 += step

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, np.int32))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cols, np.float64))
        return mesh
