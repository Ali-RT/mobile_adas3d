from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config


def apply_runtime_overrides(
    config: Dict[str, Any],
    profile: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    if profile is not None:
        if profile not in config["dataset"]["profiles"]:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available profiles: {list(config['dataset']['profiles'].keys())}"
            )

        config["dataset"]["active_profile"] = profile

    active_profile = config["dataset"]["active_profile"]

    outputs_cfg = config["outputs"]
    profile_output_dirs = outputs_cfg.get("profile_output_dirs", {})

    if active_profile in profile_output_dirs:
        output_dir = profile_output_dirs[active_profile]
    else:
        output_dir = outputs_cfg.get("output_dir", "./outputs")

    outputs_cfg["output_dir"] = output_dir
    outputs_cfg["runs_dir"] = str(Path(output_dir) / "runs")
    outputs_cfg["checkpoint_dir"] = str(Path(output_dir) / "checkpoints")
    outputs_cfg["log_dir"] = str(Path(output_dir) / "logs")
    outputs_cfg["visualization_dir"] = str(Path(output_dir) / "visualizations")

    if run_name is not None:
        config["logging"]["run_name"] = run_name

    return config


def load_runtime_config_from_args(args) -> Dict[str, Any]:
    config = load_config(args.config)

    config = apply_runtime_overrides(
        config=config,
        profile=args.profile,
        run_name=getattr(args, "run_name", None),
    )

    return config