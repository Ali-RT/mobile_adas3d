from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json
import shutil


def create_run_dir(config: Dict[str, Any], config_path: str) -> Dict[str, Path]:
    run_name = config["logging"]["run_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_id = f"{timestamp}_{run_name}"

    runs_dir = Path(config["outputs"]["runs_dir"])
    run_dir = runs_dir / run_id

    dirs = {
        "run_dir": run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "log_dir": run_dir / "logs",
        "tensorboard_dir": run_dir / "tensorboard",
        "config_dir": run_dir / "config",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    if config["logging"].get("save_config_snapshot", True):
        shutil.copy2(config_path, dirs["config_dir"] / Path(config_path).name)

        with (dirs["config_dir"] / "resolved_config.json").open("w") as f:
            json.dump(config, f, indent=2)

    return dirs