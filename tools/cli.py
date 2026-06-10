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

    return parser.parse_args()


def parse_config_arg(description: str) -> argparse.Namespace:
    """
    Backward-compatible helper for existing check scripts.
    """
    return parse_config_profile_args(description)