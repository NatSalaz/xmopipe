import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import time
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Render SMPL-X .npz files to video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────
    parser.add_argument(
        "--npz_file",
        type=str,
        required=True,
        help="Path to a .npz file OR a folder of .npz files (use with --folder)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output video",
    )
    parser.add_argument(
        "--output_file", type=str, default="output.mp4", help="Output filename"
    )

    # ── Mode flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--folder",
        action="store_true",
        help="Grid-render ALL .npz files inside --npz_file (treated as folder)",
    )
    parser.add_argument(
        "--skeleton",
        action="store_true",
        help="Skeleton/joints-only visualization (single scene)",
    )

    # ── Grid options (only used with --folder) ────────────────────────────
    parser.add_argument(
        "--spacing_x",
        type=float,
        default=5.0,
        help="Horizontal spacing between scenes on the grid (metres)",
    )
    parser.add_argument(
        "--spacing_z",
        type=float,
        default=5.0,
        help="Depth spacing between scene rows on the grid (metres)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1920x1080",
        help="Output resolution WxH, e.g. 1920x1080 or 1280x720",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--no_loop",
        action="store_true",
        help="Don't loop shorter sequences (stop them instead)",
    )
    parser.add_argument(
        "--no_labels", action="store_true", help="Hide scene-name overlays"
    )
    parser.add_argument(
        "--cam_elevation",
        type=float,
        default=0.6,
        help="Camera elevation blend: 0=horizontal, 1=top-down",
    )
    parser.add_argument(
        "--cam_distance",
        type=float,
        default=1.3,
        help="Multiply auto-computed camera distance",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_folder",
        type=str,
        default="data/smplx_models",
        help="Path to SMPL-X model folder",
    )

    args = parser.parse_args()

    # Parse resolution
    try:
        w, h = [int(x) for x in args.resolution.lower().split("x")]
    except ValueError:
        parser.error("--resolution must be WxH, e.g. 1920x1080")

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.folder:
        from render_utils.folder_render import render_folder_grid

        render_folder_grid(
            folder=args.npz_file,
            output_dir=args.output_dir,
            output_file=args.output_file,
            model_folder=args.model_folder,
            resolution=(w, h),
            fps=args.fps,
            loop=not args.no_loop,
            show_labels=not args.no_labels,
        )

    elif args.skeleton:
        from render_utils.folder_render import (
            render_multi_person_with_overlay_skeleton,
        )  # noqa (keep original import if needed)

        render_multi_person_with_overlay_skeleton(
            npz_file=args.npz_file,
            output_dir=args.output_dir,
            output_file=args.output_file,
            model_folder=args.model_folder,
        )

    else:
        from render_utils.folder_render import render_multi_person_with_overlay

        render_multi_person_with_overlay(
            npz_file=args.npz_file,
            output_dir=args.output_dir,
            output_file=args.output_file,
            model_folder=args.model_folder,
        )


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal time: {time.time() - start:.1f}s")
