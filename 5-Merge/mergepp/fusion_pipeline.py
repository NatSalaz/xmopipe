import argparse
from smoothing import *
from merge import *


def main():
    parser = argparse.ArgumentParser(
        description="Merged face and body datas and organize output by video,scene."
    )
    parser.add_argument(
        "--input_body",
        required=True,
        help="input path to the body .npz files.",
    )
    parser.add_argument(
        "--input_face",
        required=True,
        help="input path to the face .npz files.",
    )
    parser.add_argument("--output_root", required=True, help="Path to output folder.")
    parser.add_argument(
        "--no_smooth",
        dest="smooth",
        action="store_false",
        help="desactivate smoothing (Gaussian smoothin around frames)",
    )
    parser.set_defaults(smooth=True)

    args = parser.parse_args()
    process_all_files(args.input_body, args.input_face, args.output_root, args.smooth)


if __name__ == "__main__":
    main()
    print("fusion done")
