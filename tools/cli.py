import argparse


def parse_config_profile_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--config",
        type=str,
        default="configs/kitti_mobileadas3d.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Dataset/runtime profile, e.g. local_mac or colab_drive.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override.",
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override the active profile's KITTI root directory.",
    )

    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Override the active profile's split directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the active profile's training output directory.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a latest.pt checkpoint.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for inference/evaluation.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to use.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of images to process.",
    )

    parser.add_argument(
        "--image-id",
        type=str,
        default=None,
        help="Optional KITTI sample id to process, e.g. 007479.",
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Prediction score threshold.",
    )

    return parser.parse_args()


def parse_config_arg(description: str) -> argparse.Namespace:
    return parse_config_profile_args(description)
