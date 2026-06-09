from typing import Any, Dict

from models.mobile_adas3d import MobileADAS3D


def build_model(config: Dict[str, Any]) -> MobileADAS3D:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    num_classes = len(dataset_cfg["classes"])

    model = MobileADAS3D(
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        use_sparse_depth=model_cfg.get("use_sparse_depth", False),
    )

    return model