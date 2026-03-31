"""latent_canvas.py - Interactive 2D latent space visualization"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt
from matplotlib.patches import FancyArrowPatch


class LatentCanvas:
    def __init__(self, coords2d, title="Latent Space", indices=None, labels=None):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self._connect_scroll_zoom()
        self.toolbar = NavigationToolbar2QT(self.canvas, None)
        self.last_knn = None

        self.token_lines = []
        self.coords2d = coords2d
        self.latent_points = coords2d.copy()
        self.indices = indices
        self.callback = None
        self.k = 5
        if labels is not None:
            self.scatter = self.ax.scatter(
                coords2d[:, 0],
                coords2d[:, 1],
                c=labels,
                cmap="tab20",
                s=20,
                picker=True,
                alpha=0.7,
            )
        else:
            self.scatter = self.ax.scatter(
                coords2d[:, 0], coords2d[:, 1], s=5, picker=True, alpha=0.7
            )

        self.highlight = self.ax.scatter(
            [], [], s=50, c="red", edgecolors="black", linewidth=2
        )
        self.ax.set_title(title)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

        self._add_k_slider()

    def _add_k_slider(self):
        """Add K neighbors slider to toolbar"""
        self.k_label = QLabel(f"K={self.k}")
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(1)
        self.k_slider.setMaximum(10)
        self.k_slider.setValue(self.k)
        self.k_slider.setSingleStep(1)
        self.k_slider.valueChanged.connect(self.update_k)

        slider_widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel("K nearest neighbors:"))
        layout.addWidget(self.k_slider)
        layout.addWidget(self.k_label)
        slider_widget.setLayout(layout)
        self.toolbar.addWidget(slider_widget)

    def update_k(self, value):
        """Update K value from slider"""
        self.k = value
        self.k_label.setText(f"K={self.k}")

    def _on_pick(self, event):
        """Handle point picking"""
        if event.mouseevent.button != 1:
            return
        ind = event.ind[0]
        global_ind = self.indices[ind] if self.indices is not None else ind
        self.update_highlight(ind)
        if self.callback:
            self.callback(global_ind, self.last_knn)

    def _on_click(self, event):
        """Handle canvas click with KNN search"""
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        click = np.array([event.xdata, event.ydata])
        diff = self.coords2d - click
        dists = np.sum(diff**2, axis=1)

        closest = int(np.argmin(dists))
        dist_closest = float(np.sqrt(dists[closest]))

        SNAP_RADIUS = 1
        global_ind = None

        if dist_closest < SNAP_RADIUS:
            snapped_ind = closest
            self.update_highlight(snapped_ind)
            global_ind = (
                self.indices[snapped_ind] if self.indices is not None else snapped_ind
            )
        else:
            self.highlight.set_offsets([event.xdata, event.ydata])
            self.canvas.draw_idle()

        K = min(self.k, len(self.coords2d))
        knn_inds = np.argsort(dists)[:K]
        knn_dists = np.sqrt(dists[knn_inds])
        self.last_knn = list(zip(knn_inds, knn_dists))

        if self.callback:
            self.callback(global_ind, self.last_knn)

    def update_token_line(self, token_positions, token_ids, anim_frame):
        """
        Displays the used tokens and join them with a line
        """
        for line in self.token_lines:
            line.remove()
        self.token_lines = []

        if hasattr(self, "_token_texts"):
            for txt in self._token_texts:
                txt.remove()
        self._token_texts = []

        for i in range(len(token_positions) - 1):
            x0, y0 = token_positions[i]
            x1, y1 = token_positions[i + 1]

            arrow = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",  # petite flèche pleine
                mutation_scale=15,  # taille de la pointe
                color="red",
                linewidth=1.5,
                alpha=0.5,
            )
            self.ax.add_patch(arrow)
            self.token_lines.append(arrow)

        for i, (x, y) in enumerate(token_positions):
            txt = self.ax.text(
                x,
                y,
                str(token_ids[i].item()),
                color="white",
                fontsize=10,
                bbox=dict(facecolor="red", alpha=0.3, boxstyle="round,pad=0.2"),
            )
            self._token_texts.append(txt)

        self.canvas.draw_idle()

    def update_highlight(self, ind):
        """Update highlighted point"""
        self.highlight.set_offsets(self.coords2d[ind : ind + 1])
        self.canvas.draw()

    def set_callback(self, callback):
        """Set callback for point selection"""
        self.callback = callback

    def _connect_scroll_zoom(self):
        """Enable mouse wheel zoom"""

        def zoom(event):
            if event.inaxes != self.ax:
                return

            scale_factor = 1 / 1.5 if event.button == "up" else 1.5
            xdata, ydata = event.xdata, event.ydata

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            new_width = (xlim[1] - xlim[0]) * scale_factor
            new_height = (ylim[1] - ylim[0]) * scale_factor

            relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
            rely = (ydata - ylim[0]) / (ylim[1] - ylim[0])

            self.ax.set_xlim([xdata - new_width * relx, xdata + new_width * (1 - relx)])
            self.ax.set_ylim(
                [ydata - new_height * rely, ydata + new_height * (1 - rely)]
            )
            self.canvas.draw_idle()

        self.fig.canvas.mpl_connect("scroll_event", zoom)
