import numpy as np
import torch
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTabWidget,
    QStatusBar,
    QMessageBox,
    QSlider,
    QLabel,
    QPushButton,
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
import open3d as o3d

from .config import ViewerConfig, load_config
from .latent_manager import LatentManager
from .data_processor import DataProcessor
from .skeleton_renderer import SkeletonRenderer
from .latent_canvas import LatentCanvas
from data.motion_loader import DATALoader
import traceback


class LoadingThread(QThread):
    """Background thread for data loading and processing"""

    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, device, config, model_manager):
        super().__init__()
        self.device = device
        self.config = config
        self.model_manager = model_manager
        self.loader = None
        self.all_z = None
        self.all_z_full = None
        self.pca_coords = None
        self.tsne_coords = None
        self.umap_coords = None
        self.sampled_indices = None

    def run(self):
        try:
            self.loader = DATALoader(
                ViewerConfig.DATASET_TO_LOAD_NAME,
                ViewerConfig.BATCH_SIZE,
                shuffle=False,
                subsampling=ViewerConfig.SUBSAMPLING_DATA,
                deterministic=True,
                normalized=False,
                data_split="test",
            )
            self.progress.emit(20)
            self.all_z_full, self.all_z_q_full = self.model_manager.encode_dataset(
                self.loader
            )
            self.progress.emit(40)

            model_type = self.config["model_type"]

            if model_type == "vqvae":
                codebook_latent = (
                    self.model_manager.model.quantizer.codebook.detach().cpu()
                )
                latents_for_dimred = codebook_latent

            elif model_type == "rvqvae":
                quantizer = self.model_manager.model.quantizer
                original_device = self.device
                quantizer.to("cpu")
                with torch.no_grad():
                    z_q_decoded = quantizer.get_codebook_entry(self.all_z_q_full.cpu())
                quantizer.to(original_device)

                # (N, 8, 64)
                latents_for_dimred = z_q_decoded.transpose(1, 2).reshape(-1, 64 * 8)

            elif model_type == "klvae":
                latents_for_dimred = self.all_z_full.reshape(
                    -1, 8 * self.config["latent_dim"]
                )

            else:
                # By default
                latents_for_dimred = self.all_z_full

            self.progress.emit(55)
            from .data_processor import DataProcessor

            self.pca_coords, self.tsne_coords = DataProcessor.reduce_pca_tsne(
                latents_for_dimred,
                perplexity=ViewerConfig.TSNE_PERPLEXITY,
            )
            self.progress.emit(75)
            from .data_processor import run_umap_subprocess

            self.umap_coords = run_umap_subprocess(latents_for_dimred)
            self.progress.emit(95)
            self.progress.emit(100)
            self.finished.emit(True, "Loading complete")

        except Exception as e:
            error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print(error_msg)
            self.finished.emit(False, error_msg)


class ViewerApp(QMainWindow):
    """Main application window for latent space visualization"""

    def __init__(self, config_path, checkpoint=None, dataset=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = load_config(config_path)

        # Override checkpoint if provided via command line
        if checkpoint is not None:
            self.config["checkpoint"] = checkpoint
            print(f"Checkpoint overridden to: {checkpoint}")

        # Override dataset if provided via command line
        if dataset is not None:
            self.config["dataset"] = dataset
            ViewerConfig.DATASET_TO_LOAD_NAME = dataset
            print(f"Dataset overridden to: {dataset}")

        self.show_knn = True

        self.model_manager = LatentManager(self.device, self.config)
        self.model_manager.load_model(self.config["model_type"], self.config)
        self.matching_idx = 0
        self.vis = None
        self.current_joints = None
        self.anim_frame = 0
        self.skeleton_geometries = []
        self.knn_weights = None

        self.setup_ui()
        self.start_loading()

    def setup_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("VAE Latent Space Viewer")
        self.setGeometry(100, 100, 1600, 900)
        self.show_knn = True
        self.show_recon = True
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.show_recon = True
        self.toggle_recon_btn = QPushButton("Reconstruction")
        self.toggle_recon_btn.setCheckable(True)
        self.toggle_recon_btn.setChecked(True)
        self.on_toggle("show_recon", self.toggle_recon_btn, True)
        self.toggle_recon_btn.toggled.connect(
            lambda checked: self.on_toggle("show_recon", self.toggle_recon_btn, checked)
        )
        self.toggle_knn_btn = QPushButton("Originals")
        self.toggle_knn_btn.setCheckable(True)
        self.toggle_knn_btn.setChecked(True)
        self.on_toggle("show_knn", self.toggle_knn_btn, True)
        self.toggle_knn_btn.toggled.connect(
            lambda checked: self.on_toggle("show_knn", self.toggle_knn_btn, checked)
        )
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=3)

        self.side_widget = QWidget()
        self.side_layout = QVBoxLayout(self.side_widget)
        self.side_layout.addWidget(self.toggle_knn_btn)
        self.side_layout.addWidget(self.toggle_recon_btn)
        layout.addWidget(self.side_widget, stretch=1)

        self._setup_k_slider()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Loading data...")

    def _setup_k_slider(self):
        """Setup K neighbors slider"""
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(1)
        self.k_slider.setMaximum(10)
        self.k_slider.setValue(1)
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

    def start_loading(self):
        """Start background data loading"""
        self.loading_thread = LoadingThread(
            self.device, self.config, self.model_manager
        )
        self.loading_thread.finished.connect(self.on_loading_finished)
        self.loading_thread.start()

    def on_loading_finished(self, success, message):
        """Handle loading completion"""
        if not success:
            QMessageBox.critical(self, "Error", message)
            self.close()
            return

        self.loader = self.loading_thread.loader
        self.all_z = self.loading_thread.all_z
        self.all_z_full = self.loading_thread.all_z_full
        self.all_z_q_full = self.loading_thread.all_z_q_full

        self.status_bar.showMessage(message)
        self.setup_visualization(
            self.loading_thread.pca_coords,
            self.loading_thread.tsne_coords,
            self.loading_thread.umap_coords,
            self.loading_thread.sampled_indices,
        )

    def setup_visualization(
        self, pca_coords, tsne_coords, umap_coords, sampled_indices
    ):
        """Setup PCA and t-SNE visualization tabs"""
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

        umap_widget = QWidget()
        umap_layout = QVBoxLayout(umap_widget)
        self.umap_canvas = LatentCanvas(umap_coords, "UMAP", indices=sampled_indices)
        self.umap_canvas.set_callback(self.show_reconstruction)
        umap_layout.addWidget(self.umap_canvas.canvas)
        self.tabs.addTab(umap_widget, "UMAP")

        self.pca_canvas.k = self.k_slider.value()
        self.tsne_canvas.k = self.k_slider.value()
        self.umap_canvas.k = self.k_slider.value()

        self.k_slider.valueChanged.connect(
            lambda val: setattr(self.pca_canvas, "k", val)
        )
        self.k_slider.valueChanged.connect(
            lambda val: setattr(self.tsne_canvas, "k", val)
        )
        self.k_slider.valueChanged.connect(
            lambda val: setattr(self.umap_canvas, "k", val)
        )

        self.status_bar.showMessage(
            f"Ready - {self.all_z_full.shape[0]} latents loaded"
        )

    def show_reconstruction(self, ind, knn_list):
        """Reconstruct sequence from latent space point"""
        K = self.k_slider.value()

        knn_indices, knn_weights = zip(*knn_list)
        knn_indices = np.array(knn_indices)
        knn_weights = np.array(knn_weights, dtype=np.float32)
        knn_weights = knn_weights / knn_weights.sum()
        self.knn_weights = knn_weights

        # Store original skeletons
        self.skeleton_geometries = []
        for idx in knn_indices:
            if self.config["model_type"] == "vqvae":
                selected_token = idx
                print(f"Clicked token: {selected_token}")
                for i, seq_tokens in enumerate(self.all_z_q_full):
                    if int(seq_tokens[0]) == selected_token:
                        self.matching_idx = i
                        break
                print(seq_tokens.cpu().numpy().flatten())
                if self.matching_idx is not None:
                    original_sequence = self.loader.dataset[self.matching_idx]
                    print(
                        f"Animation randomly (Not really) chosen for this token: {self.matching_idx}"
                    )
                else:
                    self.matching_idx = 0
                    original_sequence = self.loader.dataset[self.matching_idx]
            else:
                original_sequence = self.loader.dataset[idx]

            # Dénormaliser la séquence originale pour la visualisation
            # original_sequence = (
            #    self.loader.dataset.inv_transform(original_sequence).cpu().numpy()
            # )
            original_sequence = original_sequence.cpu().numpy()
            original_joints = SkeletonRenderer.motion_to_joints(original_sequence)
            self.skeleton_geometries.append({"joints": original_joints, "alpha": 0.3})

        # Interpolation des latents si K > 1
        if K > 1:
            latent_vecs = self.all_z_full[knn_indices]  # (K, latent_dim)
            weights = torch.tensor(
                knn_weights, device=latent_vecs.device, dtype=latent_vecs.dtype
            )
            if self.config["model_type"] == "klvae":
                weights = weights.view(-1, 1, 1)  # (K, 1, 1)
                latent_vec_interp = (latent_vecs * weights).sum(dim=0, keepdim=True)
            else:
                weights = weights.view(-1, 1)  # (K, 1)
                latent_vec_interp = (latent_vecs * weights).sum(dim=0, keepdim=True)
        else:
            latent_vec_interp = self.all_z_full[knn_indices[0]].unsqueeze(0)

        # Decode selon le type de modèle
        if self.config["model_type"] == "vqvae":
            # IMPORTANT: decode retourne des données normalisées (input space ou model space)
            # On veut return_in_input_space=False pour avoir l'espace brut
            recon_interp = self.model_manager.decode(
                self.all_z_q_full[self.matching_idx],
                return_in_input_space=False,  # Retourne en espace brut (dénormalisé)
            )

        elif self.config["model_type"] == "rvqvae":
            indices = self.all_z_q_full[idx]  # (n, q)
            indices = indices.unsqueeze(0)  # (1, n, q)
            indices = indices.to(self.device)
            recon_interp = self.model_manager.decode(
                indices,
                return_in_input_space=False,  # Retourne en espace brut (dénormalisé)
            )

        else:
            # VAE, KLVAE, etc.
            recon_interp = self.model_manager.decode(
                latent_vec_interp,
                return_in_input_space=False,  # Retourne en espace brut (dénormalisé)
            )

        # La reconstruction est maintenant en espace brut (dénormalisé)
        # Même espace que les séquences originales après inv_transform()
        recon_np = recon_interp.squeeze(0).cpu().numpy()
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

            self.ground_geom = SkeletonRenderer.create_ground_plane()
            self.vis.add_geometry(self.ground_geom)

            ctr = self.vis.get_view_control()
            cam_pos = np.array([-0.375, -0.5, -0.75])
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_up([0.0, 1.0, 0.0])
            ctr.set_front((np.array([0.0, 0.0, 0.0]) - cam_pos))

            self.anim_timer = QTimer()
            self.anim_timer.timeout.connect(self.update_animation)
            self.anim_timer.start(33)

        self.update_skeleton()

    def on_toggle(self, attr_name, button, checked):
        setattr(self, attr_name, checked)

        # Définir la couleur selon le bouton
        if attr_name == "show_knn":
            color = "red" if checked else "white"
        elif attr_name == "show_recon":
            color = "green" if checked else "white"
        else:
            color = "white"

        button.setStyleSheet(f"background-color: {color}; color: white;")

    def update_skeleton(self):
        """Update skeleton meshes in viewer"""
        if self.current_joints is None or not self.skeleton_geometries:
            return

        if hasattr(self, "meshes_to_remove"):
            for mesh in self.meshes_to_remove:
                self.vis.remove_geometry(mesh, reset_bounding_box=False)

        self.meshes_to_remove = []

        # KNN skeletons (optionnels)
        if self.show_knn:
            for i, skel_info in enumerate(self.skeleton_geometries):
                joints = skel_info["joints"][self.anim_frame]
                color = [
                    1,
                    1 - float(self.knn_weights[i]),
                    1 - float(self.knn_weights[i]),
                ]
                mesh = SkeletonRenderer.create_skeleton_mesh(joints, color=color)
                self.vis.add_geometry(mesh, reset_bounding_box=False)
                self.meshes_to_remove.append(mesh)

        # Reconstructed skeleton (optionnel)
        if self.show_recon and self.current_joints is not None:
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
        if self.config["model_type"] == "vqvae":
            selected_token_indices = self.all_z_q_full[self.matching_idx]  # (T_tokens,)
            selected_token_indices = torch.tensor(
                [x[0] for x in selected_token_indices]
            )  # [[1],[2],[3]] ==> [1,2,3]

            token_positions = self.loading_thread.tsne_coords[
                selected_token_indices.cpu().numpy()
            ]
            self.tsne_canvas.update_token_line(
                token_positions, selected_token_indices, self.anim_frame
            )
        self.update_skeleton()
        self.vis.poll_events()
        self.vis.update_renderer()
        self.anim_frame = (self.anim_frame + 1) % len(self.current_joints)

    def closeEvent(self, event):
        """Handle window close"""
        if self.vis is not None:
            self.vis.destroy_window()
        event.accept()
