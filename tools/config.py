from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import yaml


CONFIG_ALIASES = {
    "kitti_mobileadas3d_colab.yaml": "kitti_mobileadas3d.yaml",
}


def resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)

    if path.exists():
        return path

    alias_name = CONFIG_ALIASES.get(path.name)

    if alias_name is not None:
        alias_path = path.with_name(alias_name)

        if alias_path.exists():
            return alias_path

    raise FileNotFoundError(f"Config file not found: {path}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str,
    _stack: Optional[tuple[Path, ...]] = None,
) -> Dict[str, Any]:
    path = resolve_config_path(config_path)
    path = path.resolve()
    stack = _stack or ()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Config inheritance cycle: {chain}")

    with path.open("r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    base_config = config.pop("base_config", None)
    if base_config is None:
        return config
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = load_config(str(base_path), _stack=(*stack, path))
    return _deep_merge(base, config)


def apply_runtime_overrides(
    config: Dict[str, Any],
    profile: Optional[str] = None,
    run_name: Optional[str] = None,
    dataset_root: Optional[str] = None,
    split_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if profile is not None:
        if profile not in config["dataset"]["profiles"]:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available profiles: {list(config['dataset']['profiles'].keys())}"
            )

        config["dataset"]["active_profile"] = profile

    active_profile = config["dataset"]["active_profile"]

    if dataset_root is not None:
        config["dataset"]["profiles"][active_profile]["root_dir"] = dataset_root

    if split_dir is not None:
        split_cfg = config["dataset"]["splits"]
        split_cfg.setdefault("profile_split_dirs", {})[active_profile] = split_dir

    outputs_cfg = config["outputs"]
    profile_output_dirs = outputs_cfg.get("profile_output_dirs", {})

    if output_dir is not None:
        resolved_output_dir = output_dir
    elif active_profile in profile_output_dirs:
        resolved_output_dir = profile_output_dirs[active_profile]
    else:
        resolved_output_dir = outputs_cfg.get("output_dir", "./outputs")

    outputs_cfg["output_dir"] = resolved_output_dir
    outputs_cfg["runs_dir"] = str(Path(resolved_output_dir) / "runs")
    outputs_cfg["checkpoint_dir"] = str(Path(resolved_output_dir) / "checkpoints")
    outputs_cfg["log_dir"] = str(Path(resolved_output_dir) / "logs")
    outputs_cfg["visualization_dir"] = str(Path(resolved_output_dir) / "visualizations")

    if run_name is not None:
        config["logging"]["run_name"] = run_name

    return config


def load_runtime_config_from_args(args) -> Dict[str, Any]:
    if isinstance(args, str):
        args = SimpleNamespace(config=args, profile=None, run_name=None)

    config_path = resolve_config_path(args.config)
    args.config = str(config_path)

    config = load_config(args.config)

    config = apply_runtime_overrides(
        config=config,
        profile=args.profile,
        run_name=getattr(args, "run_name", None),
        dataset_root=getattr(args, "dataset_root", None),
        split_dir=getattr(args, "split_dir", None),
        output_dir=getattr(args, "output_dir", None),
    )

    return config
