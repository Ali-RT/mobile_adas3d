import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_SUBDIRS = {
    "training/image_2": ".png",
    "training/label_2": ".txt",
    "training/calib": ".txt",
}
MANIFEST_NAME = ".mobileadas3d_stage_manifest.json"


def count_files(directory: Path, suffix: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix == suffix
    )


def directory_size_bytes(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def collect_counts(root: Path) -> dict[str, int]:
    return {
        subdir: count_files(root / subdir, suffix)
        for subdir, suffix in REQUIRED_SUBDIRS.items()
    }


def validate_counts(
    counts: dict[str, int],
    *,
    expected_count: int,
    root: Path,
    label: str,
) -> None:
    missing = [subdir for subdir in REQUIRED_SUBDIRS if not (root / subdir).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{label} is missing KITTI subdirectories under {root}: "
            + ", ".join(missing)
        )

    bad_counts = {
        subdir: count
        for subdir, count in counts.items()
        if count != expected_count
    }
    if bad_counts:
        formatted = ", ".join(
            f"{subdir}={count}" for subdir, count in bad_counts.items()
        )
        raise RuntimeError(
            f"{label} does not look like full KITTI training data. "
            f"Expected {expected_count} files in each required folder, got {formatted}."
        )


def print_counts(title: str, root: Path, counts: dict[str, int]) -> None:
    print(f"\n{title}: {root}")
    for subdir in REQUIRED_SUBDIRS:
        print(f"  {subdir}: {counts[subdir]}")


def manifest_path(destination: Path) -> Path:
    return destination / MANIFEST_NAME


def load_manifest(destination: Path) -> dict | None:
    path = manifest_path(destination)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_manifest(
    destination: Path,
    *,
    source: Path,
    expected_count: int,
    complete: bool,
    source_counts: dict[str, int],
    destination_counts: dict[str, int],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "complete": complete,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "expected_count_per_folder": expected_count,
        "required_subdirs": REQUIRED_SUBDIRS,
        "source_counts": source_counts,
        "destination_counts": destination_counts,
        "destination_size_bytes": directory_size_bytes(destination),
    }
    with manifest_path(destination).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def is_complete(destination: Path, expected_count: int) -> bool:
    manifest = load_manifest(destination)
    counts = collect_counts(destination)
    if any(counts[subdir] != expected_count for subdir in REQUIRED_SUBDIRS):
        return False
    return bool(manifest and manifest.get("complete") is True)


def make_progress_bar(description: str, total: int, initial: int):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None

    return tqdm(
        total=total,
        initial=min(initial, total),
        desc=description,
        unit="files",
        dynamic_ncols=True,
    )


def run_rsync(
    source: Path,
    destination: Path,
    *,
    suffix: str,
    expected_count: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-ah",
        "--partial",
        "--stats",
        f"{source}/",
        f"{destination}/",
    ]
    print("\nRunning:", " ".join(command))

    initial_count = count_files(destination, suffix)
    progress_bar = make_progress_bar(
        destination.name,
        total=expected_count,
        initial=initial_count,
    )
    process = subprocess.Popen(command)
    last_count = min(initial_count, expected_count)

    try:
        while process.poll() is None:
            time.sleep(1.0)
            current_count = min(count_files(destination, suffix), expected_count)
            if progress_bar is not None:
                progress_bar.update(max(0, current_count - last_count))
            elif current_count != last_count:
                print(f"  {destination.name}: {current_count}/{expected_count} files")
            last_count = current_count

        final_count = min(count_files(destination, suffix), expected_count)
        if progress_bar is not None:
            progress_bar.update(max(0, final_count - last_count))
        elif final_count != last_count:
            print(f"  {destination.name}: {final_count}/{expected_count} files")
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def copy_subdir(
    source_root: Path,
    destination_root: Path,
    subdir: str,
    *,
    expected_count: int,
) -> None:
    source = source_root / subdir
    destination = destination_root / subdir
    if not source.is_dir():
        raise FileNotFoundError(f"Source KITTI subdirectory missing: {source}")
    run_rsync(
        source,
        destination,
        suffix=REQUIRED_SUBDIRS[subdir],
        expected_count=expected_count,
    )


def stage_kitti(
    *,
    source: Path,
    destination: Path,
    expected_count: int,
    force: bool,
) -> Path:
    print("MobileADAS3D KITTI staging")
    print(f"Source:      {source}")
    print(f"Destination: {destination}")
    print(f"Expected files per required folder: {expected_count}")

    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is required in Colab for resumable KITTI staging.")

    if not source.is_dir():
        raise FileNotFoundError(f"KITTI Drive directory not found: {source}")

    source_counts = collect_counts(source)
    print_counts("Source counts", source, source_counts)
    validate_counts(
        source_counts,
        expected_count=expected_count,
        root=source,
        label="Source",
    )

    destination_counts = collect_counts(destination)
    print_counts("Current staged counts", destination, destination_counts)

    if not force and is_complete(destination, expected_count):
        print("\nLocal KITTI stage is already complete. Skipping copy.")
        return destination

    write_manifest(
        destination,
        source=source,
        expected_count=expected_count,
        complete=False,
        source_counts=source_counts,
        destination_counts=destination_counts,
    )

    for subdir in REQUIRED_SUBDIRS:
        print("\n----------------------------------------")
        print(f"Staging {subdir}")
        copy_subdir(
            source,
            destination,
            subdir,
            expected_count=expected_count,
        )

    destination_counts = collect_counts(destination)
    print_counts("Final staged counts", destination, destination_counts)
    validate_counts(
        destination_counts,
        expected_count=expected_count,
        root=destination,
        label="Destination",
    )
    write_manifest(
        destination,
        source=source,
        expected_count=expected_count,
        complete=True,
        source_counts=source_counts,
        destination_counts=destination_counts,
    )

    print(f"\nStaging manifest: {manifest_path(destination)}")
    print("KITTI staging complete.")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage KITTI from Google Drive to local Colab storage."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=7481)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run rsync even when the local staged copy already validates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_kitti(
        source=args.source,
        destination=args.destination,
        expected_count=args.expected_count,
        force=args.force,
    )


if __name__ == "__main__":
    main()
