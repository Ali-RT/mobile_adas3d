import torch

from tools.config import load_config


def main() -> None:
    config = load_config("configs/kitti_mobileadas3d.yaml")

    print("Config loaded successfully.")
    print(f"Project: {config['project_name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Classes: {config['dataset']['classes']}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()