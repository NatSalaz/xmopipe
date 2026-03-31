#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from PyQt5.QtWidgets import QApplication

from latentspace.viewer_app import ViewerApp
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """Main entry point"""
    parser = ArgumentParser(description="Motion Latent Space Viewer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (YAML)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint to load (overrides config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to load (overrides config)",
    )

    args = parser.parse_args()
    print(f"Config used: {args.config}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    if args.dataset:
        print(f"Dataset: {args.dataset}")

    app = QApplication(sys.argv)
    viewer = ViewerApp(args.config, checkpoint=args.checkpoint, dataset=args.dataset)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
