import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import time
import argparse

# from render_utils.fast_render import render_multi_person_with_overlay
# from render_utils.fast_render import render_multi_person_with_overlay_skeleton
from render_utils.fast_render_CL import render_single_frame_mesh


def main():
    parser = argparse.ArgumentParser(description="Render a .npz file to a mesh video.")
    parser.add_argument(
        "--npz_file",
        type=str,
        required=True,
        help="Path to the .npz file or directory containing .npz files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output videos",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Directory to save the output videos",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        required=True,
        help="Frame to display",
    )
    parser.add_argument(
        "--body",
        type=int,
        default=None,
        help="Body to display",
    )
    args = parser.parse_args()

    render_single_frame_mesh(
        npz_file=args.npz_file,
        output_path=args.output_dir + "/" + args.output_file,
        frame_idx=args.frame_idx,
        resolution=(2048, 2048),
        body=args.body,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"{time.time()-start_time}s taken to render.")
