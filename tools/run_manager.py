from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json
import shutil


def run_dirs_from_root(run_dir: Path) -> Dict[str, Path]:
    return {
        "run_dir": run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "log_dir": run_dir / "logs",
        "tensorboard_dir": run_dir / "tensorboard",
        "config_dir": run_dir / "config",
    }


def create_run_dir(config: Dict[str, Any], config_path: str) -> Dict[str, Path]:
    run_name = config["logging"]["run_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_id = f"{timestamp}_{run_name}"

    runs_dir = Path(config["outputs"]["runs_dir"])
    run_dir = runs_dir / run_id

    dirs = run_dirs_from_root(run_dir)

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    if config["logging"].get("save_config_snapshot", True):
        shutil.copy2(config_path, dirs["config_dir"] / Path(config_path).name)

        with (dirs["config_dir"] / "resolved_config.json").open("w") as f:
            json.dump(config, f, indent=2)

    return dirs


def resume_run_dir(checkpoint_path: str | Path) -> Dict[str, Path]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError(
            "Resume checkpoint must be inside <run>/checkpoints so the existing "
            f"run can be recovered; got {checkpoint_path}"
        )
    dirs = run_dirs_from_root(checkpoint_path.parent.parent)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
