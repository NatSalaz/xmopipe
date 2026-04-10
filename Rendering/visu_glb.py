import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import time
import argparse
from render_utils.glb_render import export_glb_animation


def main():
    parser = argparse.ArgumentParser(description="Render NPZ to animated GLB")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=60)

    args = parser.parse_args()

    output_path = os.path.join("./", args.output)

    export_glb_animation(
        npz_file=args.input,
        output_path=output_path,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"{time.time() - start_time:.2f}s taken.")
