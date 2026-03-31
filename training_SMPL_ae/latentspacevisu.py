#!/usr/bin/env python3
"""Interactive latent space viewer for KL-VAE with PyQt5 and Open3D rendering"""
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from argparse import ArgumentParser
from os.path import join as pjoin

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QProgressBar,
    QStatusBar,
    QMessageBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import open3d as o3d

from models.vae import VAE
from data.motion_loader import DATALoader
import yaml

model_manager = None
all_z = None
all_z_full = None
pca_coords = None
tsne_coords = None
sampled_indices = None


@staticmethod
def load_config(config_path: str):
    """Load config from YAML"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        print(config)
    return config


class Config:
    DATASET_NAME = "xmo"
    DATASET_TO_LOAD_NAME = "xmo"
    BATCH_SIZE = 32
    SUBSAMPLE_STEP = 10
    TSNE_PERPLEXITY = 30
    if DATASET_TO_LOAD_NAME == "xmo":
        SUBSAMPLING_DATA = 0.0003
    else:
        SUBSAMPLING_DATA = 1.00


class LatentManager:
    def __init__(self, device: torch.device, config: dict):
        """
        device: torch.device
        config: dict loaded from YAML
        """
        self.device = device
        self.config = config
        self.model = None
        self.mean = None
        self.std = None

    def load_model(self, model_name: str, config: dict):
        """Factory to create model"""
        input_dim = config["input_dim"]
        latent_dim = config["latent_dim"]

        # if model_name == "vqvae":
        #    return VQVAE(config=config)
        if model_name == "vae":
            self.model = VAE(input_dim=input_dim, latent_dim=latent_dim)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        ckpt_path = f"experiments/{config['model_type']}_{config['dataset_name']}/checkpoints/latest.pth"
        print("Checkpoint taken from:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        # Mettre les poids dans le modèle
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        print("Model weights loaded successfully.")

    def encode_dataset(self, dataloader):
        """
        Encode the full dataset into latent vectors.
        Returns:
            all_z_flat: (N*T, D)
            all_z_full: (N, T, D)
        """

        # Load normalization stats from meta_dir specified in YAML
        data_trained_dir = self.config["data_root"]
        if data_trained_dir is None:
            data_trained_dir = dataloader.dataset.data_root
        mean = (
            torch.from_numpy(np.load(f"{data_trained_dir}/Mean.npy"))
            .float()
            .to(self.device)
        )
        std = (
            torch.from_numpy(np.load(f"{data_trained_dir}/Std.npy"))
            .float()
            .to(self.device)
        )

        # Reshape to broadcast
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)

        self.mean = mean
        self.std = std

        all_z_full = []
        print(std.device)
        print(mean.device)
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:

                # ---- GPU → CPU ----
                x = batch.float().to(self.device)
                x_cpu = x.cpu()

                # ---- CPU Tensor → numpy ----
                x_np = x_cpu.numpy()

                # ---- Apply inverse transform (numpy only) ----
                x_denorm_np = dataloader.dataset.inv_transform(x_np)

                # ---- numpy → torch GPU ----
                x_denorm = torch.from_numpy(x_denorm_np).float().to(self.device)

                # ---- Normalize using dataset stats ----
                x_norm = (x_denorm - mean) / std

                # ---- Encode ----
                mu, logvar, z = self.model.encode(
                    x_norm
                )  # encode retourne (mu, logvar, z)

                # ---- Reshape pour garder la dimension séquence (B, T, latent_dim) ----
                B, T, _ = x.shape  # x.shape = (B, T, input_dim)
                z_full = mu.view(B, T, -1)

                # ---- Ramener sur CPU pour concat plus tard ----
                all_z_full.append(z_full.cpu())

        # ---- Concaténation sur CPU ----
        all_z_full = torch.cat(all_z_full, dim=0)  # (N, T, D)
        N, T, D = all_z_full.shape
        print(all_z_full.shape)
        all_z_flat = all_z_full.reshape(N * T, D)

        return all_z_flat, all_z_full

    def decode(self, z_full):
        """Decode latent vectors back into motion data"""

        self.model.eval()
        with torch.no_grad():
            z_full = z_full.to(self.device)
            recon = self.model.decode(z_full, 1, self.config["window_size"])
            recon = recon * self.std + self.mean
            return recon


class DataProcessor:
    @staticmethod
    def reduce_dimensions(latents: torch.Tensor):
        latents_np = latents.numpy()
        pca_coords = PCA(n_components=2).fit_transform(latents_np)

        n_sample = max(1, len(latents_np) // Config.SUBSAMPLE_STEP)
        idx = np.random.choice(len(latents_np), n_sample, replace=False)
        tsne_coords = TSNE(
            n_components=2, perplexity=Config.TSNE_PERPLEXITY, random_state=42
        ).fit_transform(latents_np[idx])

        return pca_coords, tsne_coords


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
        """Convert 263-dim motion representation to 3D joints with root orientation applied"""
        seq_len = recon_np.shape[0]
        joint_num = 22
        dt = 1 / 30.0

        # Racine
        root_rot_vel_y = recon_np[:, 0]  # vitesse angulaire Y
        root_lin_vel_xz = recon_np[:, 1:3]  # vitesses linéaires XZ
        root_y = recon_np[:, 3]  # hauteur Y

        # Intégration de la vitesse angulaire pour obtenir l'orientation
        root_orient_y = np.cumsum(root_rot_vel_y * dt)
        # Optionnel : ajouter un offset pour tourner le personnage
        # root_orient_y += np.pi  # pour 180° rotation

        cos_y = np.cos(root_orient_y)
        sin_y = np.sin(root_orient_y)

        # Calcul de la trajectoire globale
        vx = root_lin_vel_xz[:, 0] * cos_y - root_lin_vel_xz[:, 1] * sin_y
        vz = root_lin_vel_xz[:, 0] * sin_y + root_lin_vel_xz[:, 1] * cos_y
        root_pos_xz = np.cumsum(np.stack([vx, vz], axis=-1), axis=0)
        root_pos = np.concatenate(
            [root_pos_xz[:, 0:1], root_y[:, None], root_pos_xz[:, 1:2]], axis=-1
        )

        # Joints locaux
        ric_start = 4
        ric_end = ric_start + (joint_num - 1) * 3
        ric_local = recon_np[:, ric_start:ric_end].reshape(seq_len, joint_num - 1, 3)

        # Appliquer la rotation Y de la racine aux joints
        ric_global = ric_local.copy()
        x = ric_global[:, :, 0]
        z = ric_global[:, :, 2]
        ric_global[:, :, 0] = x * cos_y[:, None] - z * sin_y[:, None]
        ric_global[:, :, 2] = x * sin_y[:, None] + z * cos_y[:, None]

        # Translation par la position racine
        ric_global[:, :, 0] += root_pos[:, 0:1]
        ric_global[:, :, 2] += root_pos[:, 2:3]

        # Assemblage final
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


from PyQt5.QtWidgets import QSlider, QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt


class LatentCanvas:
    def __init__(self, coords2d, title="Latent Space", indices=None):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

        # --- Matplotlib figure ---
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self._connect_scroll_zoom()

        # --- Toolbar Qt ---
        self.toolbar = NavigationToolbar2QT(self.canvas, None)
        self.last_knn = None
        # --- Données ---
        self.coords2d = coords2d
        self.latent_points = coords2d.copy()
        self.indices = indices
        self.callback = None
        self.k = 5  # valeur par défaut

        # --- Scatter ---
        self.scatter = self.ax.scatter(
            coords2d[:, 0], coords2d[:, 1], s=5, picker=True, alpha=0.7
        )
        self.highlight = self.ax.scatter(
            [], [], s=50, c="red", edgecolors="black", linewidth=2
        )
        self.ax.set_title(title)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

        # --- Slider Qt intégré dans la toolbar ---
        self.k_label = QLabel(f"K={self.k}")
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(1)
        self.k_slider.setMaximum(10)
        self.k_slider.setValue(self.k)
        self.k_slider.setSingleStep(1)
        self.k_slider.valueChanged.connect(self.update_k)

        # Widget container pour la toolbar
        slider_widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel("K nearest neighbors:"))
        layout.addWidget(self.k_slider)
        layout.addWidget(self.k_label)
        slider_widget.setLayout(layout)
        self.toolbar.addWidget(slider_widget)

    def update_k(self, value):
        self.k = value
        self.k_label.setText(f"K={self.k}")
        print(f"[Slider] Nouveau K = {self.k}")

    def _on_pick(self, event):
        if event.mouseevent.button != 1:
            return
        ind = event.ind[0]
        global_ind = self.indices[ind] if self.indices is not None else ind
        self.update_highlight(ind)
        if self.callback:
            self.callback(global_ind, self.last_knn)

    def _on_click(self, event):
        if event.button != 1:
            return

        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None:
            return

        click = np.array([x_click, y_click])

        diff = self.coords2d - click
        dists = np.sum(diff**2, axis=1)

        closest = int(np.argmin(dists))
        dist_closest = float(np.sqrt(dists[closest]))

        SNAP_RADIUS = 1
        global_ind = None  # par défaut
        if dist_closest < SNAP_RADIUS:
            snapped_ind = closest
            self.update_highlight(snapped_ind)
            global_ind = (
                self.indices[snapped_ind] if self.indices is not None else snapped_ind
            )
        else:
            self.highlight.set_offsets([x_click, y_click])
            self.canvas.draw_idle()
            snapped_ind = None

        K = min(self.k, len(self.coords2d))
        knn_inds = np.argsort(dists)[:K]
        knn_dists = np.sqrt(dists[knn_inds])

        self.last_knn = list(zip(knn_inds, knn_dists))

        if self.callback:
            # renvoie le point sélectionné (None si pas de snap) et les KNN
            self.callback(global_ind, self.last_knn)
        print(f"[KNN] K={K} | {self.last_knn}")

    def update_highlight(self, ind):
        self.highlight.set_offsets(self.coords2d[ind : ind + 1])
        self.canvas.draw()

    def set_callback(self, callback):
        self.callback = callback

    def _connect_scroll_zoom(self):
        def zoom(event):
            if event.inaxes != self.ax:
                return

            # Sens du zoom
            scale_factor = 1 / 1.5 if event.button == "up" else 1.5

            xdata = event.xdata  # position souris en x
            ydata = event.ydata  # position souris en y

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            # Nouveau range
            new_width = (xlim[1] - xlim[0]) * scale_factor
            new_height = (ylim[1] - ylim[0]) * scale_factor

            # Ratio pour centrer EXACTEMENT sur la souris
            relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
            rely = (ydata - ylim[0]) / (ylim[1] - ylim[0])

            self.ax.set_xlim([xdata - new_width * relx, xdata + new_width * (1 - relx)])
            self.ax.set_ylim(
                [ydata - new_height * rely, ydata + new_height * (1 - rely)]
            )

            self.canvas.draw_idle()

        # Branche la molette
        self.fig.canvas.mpl_connect("scroll_event", zoom)


class ViewerApp(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vis = None
        self.current_joints = None
        self.anim_frame = 0
        self.setup_ui()
        self.args = args
        self.start_loading()

    def setup_ui(self):
        self.setWindowTitle("KL-VAE Latent Space Viewer")
        self.setGeometry(100, 100, 1600, 900)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=3)

        self.side_widget = QWidget()
        self.side_layout = QVBoxLayout(self.side_widget)
        layout.addWidget(self.side_widget, stretch=1)

        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(1)
        self.k_slider.setMaximum(10)  # ou max adapté à ton dataset
        self.k_slider.setValue(3)
        self.k_slider.setSingleStep(1)

        self.k_label = QLabel(f"K = {self.k_slider.value()}")
        self.k_slider.valueChanged.connect(
            lambda val: self.k_label.setText(f"K = {val}")
        )

        slider_container = QWidget()
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.k_label)
        slider_layout.addWidget(self.k_slider)
        slider_container.setLayout(slider_layout)
        self.side_layout.addWidget(slider_container)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Loading data...")

    def start_loading(self):

        opt = load_config(f"{self.args.config}")
        global model_manager
        model_manager = LatentManager(self.device, opt)
        model_manager.load_model(opt["model_type"], opt)

        self.loading_thread = LoadingThread(self.device)
        self.loading_thread.finished.connect(self.on_finished)
        self.loading_thread.start()

    def on_finished(self, success, message):
        if not success:
            QMessageBox.critical(self, "Error", message)
            self.close()
            return
        self.status_bar.showMessage(message)
        self.setup_visualization()

    def setup_visualization(self):
        global all_z, all_z_full, pca_coords, tsne_coords, sampled_indices

        pca_widget = QWidget()
        pca_layout = QVBoxLayout(pca_widget)
        self.pca_canvas = LatentCanvas(pca_coords, "PCA", indices=None)
        self.pca_canvas.set_callback(self.show_reconstruction)
        pca_layout.addWidget(self.pca_canvas.canvas)
        self.tabs.addTab(pca_widget, "PCA")

        tsne_widget = QWidget()
        tsne_layout = QVBoxLayout(tsne_widget)
        self.tsne_canvas = LatentCanvas(tsne_coords, "t-SNE", indices=sampled_indices)
        self.tsne_canvas.set_callback(self.show_reconstruction)
        tsne_layout.addWidget(self.tsne_canvas.canvas)
        self.tabs.addTab(tsne_widget, "t-SNE")
        self.pca_canvas.k = self.k_slider.value()
        self.tsne_canvas.k = self.k_slider.value()

        # Mettre à jour k quand le slider change
        self.k_slider.valueChanged.connect(
            lambda val: setattr(self.pca_canvas, "k", val)
        )
        self.k_slider.valueChanged.connect(
            lambda val: setattr(self.tsne_canvas, "k", val)
        )
        self.status_bar.showMessage(f"Ready - {all_z.shape[0]} latents loaded")

    def show_reconstruction(self, ind, knn_list):
        """
        Reconstruit la séquence cliquée et ses K plus proches voisins.
        Stocke tous les squelettes dans self.skeleton_geometries avec transparence.
        """
        global all_z_full, full_loader, model_manager

        K = self.k_slider.value()  # récupérer la valeur du slider

        N, T, D = all_z_full.shape
        print(f"Point cliqué : {ind}")
        print(f"KNN : {knn_list}")

        knn_indices, knn_weights = zip(*knn_list)
        knn_indices = np.array(knn_indices)
        knn_weights = np.array(knn_weights, dtype=np.float32)
        knn_weights = knn_weights / knn_weights.sum()  # normaliser les poids
        self.knn_weights = knn_weights

        self.skeleton_geometries = []
        for i, idx in enumerate(knn_indices):
            original_sequence = full_loader.dataset[idx // T]
            original_sequence = (
                (original_sequence * full_loader.dataset.std + full_loader.dataset.mean)
                .cpu()
                .numpy()
            )

            original_joints = SkeletonRenderer.motion_to_joints(original_sequence)
            self.skeleton_geometries.append({"joints": original_joints, "alpha": 0.3})

        if K > 1:
            latent_vecs = all_z_full[knn_indices // T]  # shape [K, T, D]
            latent_vecs = latent_vecs.to(model_manager.std.device)
            weights = torch.tensor(
                knn_weights, device=latent_vecs.device, dtype=latent_vecs.dtype
            )
            weights = weights.view(-1, 1, 1)  # broadcast sur [K, T, D]
            latent_vec_interp = (latent_vecs * weights).sum(dim=0, keepdim=True)
        else:
            latent_vec_interp = all_z_full[knn_indices[0] // T].unsqueeze(0)
        recon_interp = model_manager.decode(latent_vec_interp)
        recon_np = recon_interp.squeeze(0).cpu().numpy()
        print(type(recon_np))
        self.current_joints = SkeletonRenderer.motion_to_joints(recon_np)
        self.init_open3d_viewer()

    def init_open3d_viewer(self):
        """Initialize or update Open3D visualizer"""
        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="Skeleton Animation", width=800, height=600, visible=True
            )

            opt = self.vis.get_render_option()
            opt.background_color = np.asarray([0.1, 0.1, 0.1])
            opt.light_on = True

            self.skeleton_geometries = None
            self.current_joints = None
            self.ground_geom = self._create_ground()
            self.vis.add_geometry(self.ground_geom)
            ctr = self.vis.get_view_control()
            cam_params = ctr.convert_to_pinhole_camera_parameters()
            cam_pos = np.array([-0.375, -0.5, -0.75])  # X, Y, Z
            ctr.set_lookat([0.0, 0.0, 0.0])  # point cible
            ctr.set_up([0.0, 1.0, 0.0])  # vecteur "up" de la caméra
            ctr.set_front((np.array([0.0, 0.0, 0.0]) - cam_pos))
            self.anim_timer = QTimer()
            self.anim_timer.timeout.connect(self.update_animation)
            self.anim_timer.start(33)

        self.update_skeleton()

    def _create_ground(self):
        """Create checkered ground plane"""
        size, tile = 7.0, 2
        N = max(1, int(round(size / tile)))
        step = size / N
        half = size * 0.5
        verts, tris, cols = [], [], []
        vid = 0

        y_ground = 0
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

                cols.extend(
                    [(0.1, 0.5, 0.9)] * 4
                    if (ix + iz) % 2 == 0
                    else [(0.2, 0.7, 1.0)] * 4
                )
                vid += 4
                x0 += step
            z0 += step

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, np.int32))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cols, np.float64))
        return mesh

    def update_skeleton(self):
        if self.current_joints is None or not hasattr(self, "skeleton_geometries"):
            return
        K = self.k_slider.value()  # récupérer la valeur du slider

        # Supprime les anciens meshes du viewer
        if hasattr(self, "meshes_to_remove"):
            for mesh in self.meshes_to_remove:
                self.vis.remove_geometry(mesh, reset_bounding_box=False)

        self.meshes_to_remove = []

        # Squelettes KNN (transparents)
        for i, skel_info in enumerate(self.skeleton_geometries):
            joints = skel_info["joints"][self.anim_frame]  # joints à l'instant t
            mesh = SkeletonRenderer.create_skeleton_mesh(
                joints,
                color=[
                    1,
                    1 - float(self.knn_weights[i]),
                    1 - float(self.knn_weights[i]),
                ],
            )
            self.vis.add_geometry(mesh, reset_bounding_box=False)
            self.meshes_to_remove.append(mesh)

        # Squelette reconstruit (rouge)
        joints_current = self.current_joints[self.anim_frame]
        mesh_current = SkeletonRenderer.create_skeleton_mesh(
            joints_current, color=[0.0, 1.0, 0]
        )
        self.vis.add_geometry(mesh_current, reset_bounding_box=False)
        self.meshes_to_remove.append(mesh_current)

    def update_animation(self):
        """Animation loop callback"""
        if self.current_joints is None:
            return

        self.update_skeleton()
        self.vis.poll_events()
        self.vis.update_renderer()
        self.anim_frame = (self.anim_frame + 1) % len(self.current_joints)

    def closeEvent(self, event):
        if self.vis is not None:
            self.vis.destroy_window()
        event.accept()


class LoadingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, device):
        super().__init__()
        self.device = device

    def run(self):
        global model_manager, all_z, all_z_full, pca_coords, tsne_coords, sampled_indices, full_loader
        try:
            full_loader = DATALoader(
                Config.DATASET_TO_LOAD_NAME,
                Config.BATCH_SIZE,
                shuffle=False,
                subsampling=Config.SUBSAMPLING_DATA,
                deterministic=True,
            )
            self.progress.emit(30)

            all_z, all_z_full = model_manager.encode_dataset(full_loader)
            self.progress.emit(70)

            n_sample = max(1, all_z.shape[0] // Config.SUBSAMPLE_STEP)
            sampled_indices = np.random.choice(all_z.shape[0], n_sample, replace=False)
            latents_sampled = all_z[sampled_indices]
            pca_coords, tsne_coords = DataProcessor.reduce_dimensions(latents_sampled)

            self.progress.emit(100)
            self.finished.emit(True, "Loading complete")
        except Exception as e:
            self.finished.emit(False, str(e))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config (with .yaml)",
    )
    sys.argv = ["latentspacevisu.py", "--config", "configs/xmo/vae.yaml"]

    args = parser.parse_args()
    print("Config used:", args.config)

    opt = load_config(args.config)
    app = QApplication(sys.argv)
    viewer = ViewerApp(args)
    viewer.show()
    sys.exit(app.exec_())
