from pathlib import Path
from typing import Dict, Any
import csv
import json


class MetricsLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_header_written = self.csv_path.exists()

    def log(self, metrics: Dict[str, Any]) -> None:
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        fieldnames = list(metrics.keys())

        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True

            writer.writerow(metrics)