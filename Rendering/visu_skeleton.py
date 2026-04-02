import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import time
import argparse

from render_utils.scene_render import render_multi_person_with_overlay_skeleton


def main():
    parser = argparse.ArgumentParser(description="Render a .npz file to a skeleton video.")
    parser.add_argument("--npz_file", type=str, required=True, help="Path to the .npz file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output video")
    parser.add_argument("--output_file", type=str, default="output_skeleton.mp4", help="Output filename")
    parser.add_argument("--no_emotion", action="store_true", help="Disable emotion overlay")
    parser.add_argument("--follow_0", action="store_true", help="Camera follows person 0")
    parser.add_argument("--from_above", action="store_true", help="Top-down camera view")
    parser.add_argument("--text_overlay", type=str, default=None, help="Text to display on video")

    args = parser.parse_args()

    render_multi_person_with_overlay_skeleton(
        npz_file=args.npz_file,
        output_dir=args.output_dir,
        output_file=args.output_file,
        emotion=not args.no_emotion,
        follow_0=args.follow_0,
        from_above=args.from_above,
        text_overlay=args.text_overlay,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"{time.time()-start_time}s taken to render.")
