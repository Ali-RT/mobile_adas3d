import torch

from tools.cli import parse_config_arg
from tools.config import load_runtime_config_from_args


def main() -> None:
    args = parse_config_arg("Check MobileADAS3D setup")
    config = load_runtime_config_from_args(args)

    print("Config loaded successfully.")
    print(f"Using config: {args.config}")
    print(f"Project: {config['project_name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Active profile: {config['dataset']['active_profile']}")
    print(f"Classes: {config['dataset']['classes']}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
