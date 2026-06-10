import torch

from tools.config import load_runtime_config_from_args


def main() -> None:
    config = load_runtime_config_from_args("configs/kitti_mobileadas3d.yaml")

    print("Config loaded successfully.")
    print(f"Project: {config['project_name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Classes: {config['dataset']['classes']}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()