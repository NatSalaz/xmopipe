import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import time
import argparse

# from render_utils.fast_render import render_multi_person_with_overlay
# from render_utils.fast_render import render_multi_person_with_overlay_skeleton
from render_utils.scene_render import (
    render_multi_person_with_overlay,
    render_multi_person_with_overlay_skeleton,
)


def main():
    parser = argparse.ArgumentParser(description="Render a .npz file to a mesh video.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the .npz file or directory containing .npz files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the output videos",
    )
    parser.add_argument(
        "--skeleton",
        action="store_true",
        help="Joints only visualization",
    )

    args = parser.parse_args()

    if args.skeleton == True:
        render_multi_person_with_overlay_skeleton(
            npz_file=args.input,
            output_dir=".",
            output_file=args.output,
        )
    else:
        render_multi_person_with_overlay(
            npz_file=args.input,
            output_dir=".",
            output_file=args.output,
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"{time.time()-start_time}s taken to render.")
