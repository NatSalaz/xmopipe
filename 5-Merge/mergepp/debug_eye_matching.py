import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_eye_points(
    nose_avg,
    leye_avg,
    reye_avg,
    vitpose,
    frame_idx=0,
):
    """
    Visualize eye + nose points used in compute_cost.
    Red   = face model (avg)
    lime  = body model (ViTPose)
    """

    nb_frames = min(
        nose_avg.shape[0],
        vitpose.shape[0],
    )
    frame_idx = min(frame_idx, nb_frames - 1)

    # Face model (mediapipe)
    face_points = np.stack(
        [
            nose_avg[frame_idx],
            leye_avg[frame_idx],
            reye_avg[frame_idx],
        ],
        axis=0,
    )

    # Body model (ViTPose)
    body_points = np.stack(
        [
            vitpose[frame_idx, 0, :2],  # nose
            vitpose[frame_idx, 2, :2],  # left eye
            vitpose[frame_idx, 1, :2],  # right eye
        ],
        axis=0,
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(
        face_points[:, 0],
        face_points[:, 1],
        c="red",
        label="Face model",
        s=80,
        alpha=0.5,
    )
    plt.scatter(
        body_points[:, 0],
        body_points[:, 1],
        c="lime",
        label="Body model (ViTPose)",
        s=80,
    )

    for i in range(3):
        plt.plot(
            [face_points[i, 0], body_points[i, 0]],
            [face_points[i, 1], body_points[i, 1]],
            "k--",
            alpha=0.3,
        )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"Eye / Nose matching – frame {frame_idx}")
    plt.show()


def visualize_face_vs_all_bodies_with_skeleton(
    image_path,
    nose_avg,
    leye_avg,
    reye_avg,
    vitpose_all,
    frame_idx=0,
):
    """
    FINAL DEBUG VIS:
    - Red    : face (mediapipe avg)
    - lime   : ViTPose skeletons
    - lime++ : ViTPose face joints
    - Yellow : distances used in compute_cost
    """

    img = Image.open(image_path)

    plt.figure(figsize=(9, 9))
    plt.imshow(img)
    plt.axis("off")
    face_pts = np.stack(
        [
            nose_avg[frame_idx],
            leye_avg[frame_idx],
            reye_avg[frame_idx],
        ],
        axis=0,
    )

    plt.scatter(
        face_pts[:, 0],
        face_pts[:, 1],
        c="red",
        s=50,
        label="Face (avg)",
        zorder=10,
        alpha=0.5,
    )
    face_center = face_pts.mean(axis=0)
    plt.text(
        face_center[0],
        face_center[1] - 120,
        "Face 0",
        color="red",
        fontsize=12,
        ha="center",
        va="center",
        zorder=11,
        bbox=dict(
            facecolor="black",
            edgecolor="none",
            alpha=0.7,
            boxstyle="round,pad=0.2",
        ),
    )
    skeleton_edges = [
        (5, 7),
        (7, 9),  # left arm
        (6, 8),
        (8, 10),  # right arm
        (11, 13),
        (13, 15),  # left leg
        (12, 14),
        (14, 16),  # right leg
        (5, 6),  # shoulders
        (11, 12),  # hips
        (5, 11),
        (6, 12),  # torso
        (0, 1),
        (0, 2),  # face
    ]
    body_i = -1
    for body_id, vp in vitpose_all.items():
        pts = vp[frame_idx, :, :2]
        body_i += 1
        for i, j in skeleton_edges:
            if i < pts.shape[0] and j < pts.shape[0]:
                plt.plot(
                    [pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    color="lime",
                    alpha=0.5,
                    linewidth=2.0,
                    zorder=2,
                )
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            c="lime",
            s=15,
            alpha=0.4,
            zorder=2,
        )
        body_face_pts = np.stack(
            [
                pts[0],  # nose
                pts[2],  # left eye
                pts[1],  # right eye
            ],
            axis=0,
        )

        plt.scatter(
            body_face_pts[:, 0],
            body_face_pts[:, 1],
            c="lime",
            s=60,
            alpha=0.5,
            zorder=6,
        )
        for i, j in [(0, 1), (0, 2)]:
            plt.plot(
                [body_face_pts[i, 0], body_face_pts[j, 0]],
                [body_face_pts[i, 1], body_face_pts[j, 1]],
                color="lime",
                alpha=0.5,
                linewidth=3,
                zorder=7,
            )
        # Yellow line shwoing the distance between each points
        for i in range(3):
            plt.plot(
                [face_pts[i, 0], body_face_pts[i, 0]],
                [face_pts[i, 1], body_face_pts[i, 1]],
                "y--",
                linewidth=3,
                alpha=1.0,
                zorder=6,
            )
        center = pts.mean(axis=0)
        plt.text(
            center[0],
            center[1],
            f"Body {body_i}",
            color="lime",
            fontsize=12,
            ha="center",
            va="center",
            zorder=9,
            bbox=dict(
                facecolor="black",
                edgecolor="none",
                alpha=0.7,
                boxstyle="round,pad=0.2",
            ),
        )

    plt.title(f"Frame {frame_idx} – Face ↔ ALL ViTPose skeletons")
    plt.show()


from matplotlib.patches import Rectangle


def visualize_bodies_bbox_and_skeleton(
    image_path,
    vitpose_all,
    frame_idx=0,
    bbox_key="bbox_xyxy",
):
    """
    Visualize:
    - Bounding box of each body
    - Full ViTPose skeleton

    vitpose_all: dict {
        body_id: {
            "vitpose": (T, K, 3),
            "bbox_xyxy": (T, 4)
        }
    }
    """

    img = Image.open(image_path)

    plt.figure(figsize=(9, 9))
    plt.imshow(img)
    plt.axis("off")

    # ViTPose skeleton edges (COCO-style, same as before)
    skeleton_edges = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (5, 6),
        (11, 12),
        (5, 11),
        (6, 12),
        (0, 1),
        (0, 2),
    ]

    body_i = -1
    for body_id, body in vitpose_all.items():
        body_i += 1

        vp = body["vitpose"]
        if frame_idx >= vp.shape[0]:
            continue

        pts = vp[frame_idx, :, :2]

        # ---------- Bounding box ----------
        if bbox_key in body and body[bbox_key] is not None:
            bbox = body[bbox_key][frame_idx]  # x1, y1, x2, y2
            x1, y1, x2, y2 = bbox

            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
                alpha=0.8,
                zorder=6,
            )
            plt.gca().add_patch(rect)

            # Label body
            plt.text(
                x1,
                y1 - 5,
                f"Body {body_i}",
                color="lime",
                fontsize=12,
                ha="left",
                va="bottom",
                zorder=7,
                bbox=dict(
                    facecolor="black",
                    edgecolor="none",
                    alpha=0.7,
                    boxstyle="round,pad=0.2",
                ),
            )

        # ---------- Skeleton ----------
        for i, j in skeleton_edges:
            if i < pts.shape[0] and j < pts.shape[0]:
                plt.plot(
                    [pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    color="lime",
                    alpha=0.4,
                    linewidth=2,
                    zorder=4,
                )

        # Joints
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            c="lime",
            s=10,
            alpha=0.4,
            zorder=4,
        )

    plt.title(f"Frame {frame_idx} – Bounding boxes + ViTPose skeletons")
    plt.show()


import os
from matplotlib.patches import Rectangle


def visualize_bodies_bbox_and_skeleton_folder(
    image_dir,
    vitpose_all,
    frame_stride=1,
    image_exts=(".jpg", ".png"),
    output_dir="./output/",
):
    """
    Visualize bounding boxes + ViTPose skeletons over a folder of images.

    image_dir     : folder containing frames
    vitpose_all   : dict {
        body_id: {
            "vitpose": (T, K, 3),
            "bbox_xyxy": (T, 4)
        }
    }
    frame_stride  : show every N-th frame
    """

    image_paths = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(image_exts)
        ]
    )

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    # Skeleton edges (same as before)
    skeleton_edges = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (5, 6),
        (11, 12),
        (5, 11),
        (6, 12),
        (0, 1),
        (0, 2),
    ]
    frame_id = -frame_stride

    for frame_idx, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        plt.figure(figsize=(9, 9))
        plt.imshow(img)
        plt.axis("off")

        body_i = -1
        frame_id += frame_stride
        for body_id, body in vitpose_all.items():

            body_i += 1

            vp = body["vitpose"]
            if frame_idx >= vp.shape[0]:
                continue

            print(frame_id)
            pts = vp[frame_id, :, :2]

            # ---------- Bounding box ----------
            bbox = body.get("bbox_xyxy", None)
            if bbox is not None and frame_id < bbox.shape[0]:
                x1, y1, x2, y2 = bbox[frame_id]

                rect = Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                    alpha=0.8,
                    zorder=6,
                )
                plt.gca().add_patch(rect)

                plt.text(
                    x1,
                    y1 - 5,
                    f"Body {body_i}",
                    color="lime",
                    fontsize=12,
                    ha="left",
                    va="bottom",
                    zorder=7,
                    bbox=dict(
                        facecolor="black",
                        edgecolor="none",
                        alpha=0.7,
                        boxstyle="round,pad=0.2",
                    ),
                )

            # ---------- Skeleton ----------
            for i, j in skeleton_edges:
                if i < pts.shape[0] and j < pts.shape[0]:
                    plt.plot(
                        [pts[i, 0], pts[j, 0]],
                        [pts[i, 1], pts[j, 1]],
                        color="lime",
                        alpha=0.4,
                        linewidth=2,
                        zorder=4,
                    )

            plt.scatter(
                pts[:, 0],
                pts[:, 1],
                c="lime",
                s=10,
                alpha=0.4,
                zorder=4,
            )

        out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()
