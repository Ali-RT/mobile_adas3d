"""Local-first I/O wrapper around the unchanged frozen V0 evaluator.

One dev-only input capsule, local evaluation/logs, occasional archived progress.
No quota bypass, automatic remount, model/metric changes or held-out reads.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_parking_v0_dev as v0
from scripts import parking_edge_common as common

REVISION = "PARKING-V0-LOCAL-FIRST-2026-10-29-r2"
EVALUATOR_SHA = "f420a4c3b40a97dd6fddfea2a21ef33dfad49a0f9881b226e7df76427ee37540"
MAX_CAPSULE_BYTES = 250_000_000


def read_once(path):
    try:
        return Path(path).read_bytes()
    except OSError as error:
        if error.errno == 5:
            raise RuntimeError(f"Input/output error reading {path}. Drive quota or mount failure is possible, not confirmed. "
                "Stop repeated retries. Preserve the files; wait for quota recovery if Colab reports a quota, "
                "or use a previously downloaded capsule on local disk. No automatic retry/remount was attempted.") from error
        raise


def stored_zip(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(members.items()):
            v0.safe_name(name)
            archive.writestr(zipfile.ZipInfo(name, (2026, 10, 29, 0, 0, 0)), raw)
    return stream.getvalue()


def read_archive(path):
    if Path(path).stat().st_size > MAX_CAPSULE_BYTES:
        raise ValueError("Unexpectedly large portable archive")
    raw = read_once(path)
    if len(raw) > MAX_CAPSULE_BYTES:
        raise ValueError("Unexpectedly large portable archive")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or len(names) > 64
                or sum(x.file_size for x in archive.infolist()) > MAX_CAPSULE_BYTES):
            raise ValueError("Oversized or duplicate portable archive inventory")
        for name in names: v0.safe_name(name)
        if archive.testzip() is not None: raise ValueError("Portable archive CRC failed")
        members = {name: archive.read(name) for name in names}
    return raw, members


def copy_once(source, destination, expected_sha=None):
    """One sequential 4 MiB stream, atomic local/Drive publication.

    A complete cached destination is reused without opening the source. An
    interrupted transfer leaves no completed file and restarts that file only.
    """
    source, destination = Path(source), Path(destination)
    if destination.is_symlink(): raise ValueError("Preserve symlinked transfer destination")
    if destination.exists():
        if expected_sha is None or common.file_sha(destination) != expected_sha:
            raise ValueError("Preserve conflicting cached transfer")
        return expected_sha
    if source.stat().st_size > MAX_CAPSULE_BYTES:
        raise ValueError("Transfer is too large for this bounded dev-only workflow; do not copy a full dataset archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        with source.open("rb") as incoming, tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".v0_transfer_", delete=False) as outgoing:
            temporary = Path(outgoing.name)
            while block := incoming.read(4 * 1024 * 1024):
                outgoing.write(block); digest.update(block)
        actual = digest.hexdigest()
        if expected_sha is not None and actual != expected_sha:
            raise ValueError("Transfer hash differs; no completed destination was published")
        if destination.exists(): raise ValueError("Another writer published this transfer; stop concurrent notebooks")
        temporary.replace(destination)
        return actual
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def capsule_members(manifest, bundle_raw, shard_reader, config):
    inventory = v0.dev_inventory(manifest, config)
    if common.sha(bundle_raw) != config["adapter_bundle_sha256"]:
        raise ValueError("Frozen V0 checkpoint bundle differs")
    members = {"benchmark/benchmark_manifest.json": common.encoded(manifest),
        "model/parking_shared_adapter_results.zip": bundle_raw}
    for i, (shard, _) in enumerate(inventory, 1):
        raw = shard_reader(shard["file"])
        if len(raw) != shard["size_bytes"] or common.sha(raw) != shard["sha256"]:
            raise ValueError("Frozen development shard differs during packaging")
        members["benchmark/"+shard["file"]] = raw
        print(f"Input capsule: verified dev archive {i}/{len(inventory)}", flush=True)
    members["input_inventory.json"] = common.encoded({"revision": REVISION,
        "benchmark_manifest_sha256": manifest["manifest_sha256"],
        "adapter_bundle_sha256": config["adapter_bundle_sha256"],
        "files": {name: {"sha256": common.sha(raw), "size_bytes": len(raw)} for name, raw in members.items()},
        "dev_images": config["expected_images"], "heldout_shard_bodies_included": 0})
    return members


def inspect_capsule(path, config):
    raw, members = read_archive(path)
    manifest = json.loads(members["benchmark/benchmark_manifest.json"])
    inventory = v0.dev_inventory(manifest, config)
    index = json.loads(members["input_inventory.json"])
    expected = {"benchmark/benchmark_manifest.json", "model/parking_shared_adapter_results.zip"}
    expected |= {"benchmark/"+s["file"] for s, _ in inventory}
    if (set(members) != expected | {"input_inventory.json"} or set(index["files"]) != expected
            or index["benchmark_manifest_sha256"] != manifest["manifest_sha256"]
            or index["adapter_bundle_sha256"] != config["adapter_bundle_sha256"]
            or index["dev_images"] != config["expected_images"] or index["heldout_shard_bodies_included"] != 0):
        raise ValueError("Capsule has missing/unexpected inputs or another benchmark")
    hashes = {"benchmark/"+s["file"]: s["sha256"] for s, _ in inventory}
    hashes["model/parking_shared_adapter_results.zip"] = config["adapter_bundle_sha256"]
    for name in expected:
        payload = members[name]
        if index["files"][name] != {"sha256": common.sha(payload), "size_bytes": len(payload)} or (name in hashes and common.sha(payload) != hashes[name]):
            raise ValueError("Capsule member bytes differ from the frozen inputs")
    return members, {"revision": REVISION, "sha256": common.sha(raw), "size_bytes": len(raw),
        "benchmark_manifest_sha256": manifest["manifest_sha256"], "dev_images": config["expected_images"],
        "dev_shards": len(inventory), "heldout_shard_bodies_included": 0}


def stage(args):
    config = v0.configuration()
    if str(args.inputs.resolve()).startswith("/content/drive/") or str(args.local_archive.resolve()).startswith("/content/drive/"):
        raise ValueError("Keep staged inputs and the local capsule on /content, not mounted Drive")
    if args.local_archive.resolve().is_relative_to(args.inputs.resolve()): raise ValueError("Keep the capsule outside the materialized input folder")
    publish = False
    if not args.local_archive.exists():
        for source in [args.benchmark, args.bundle, args.archive]:
            if source is not None and any(local.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(local.resolve())
                                          for local in [args.inputs, args.local_archive]):
                raise ValueError("Keep immutable sources, local archive and materialized inputs separate")
        if args.archive is not None and args.archive.exists():
            print("Copying one input capsule to local disk; no source dataset scan", flush=True)
            copy_once(args.archive, args.local_archive)
        else:
            if args.benchmark is None or args.bundle is None:
                raise FileNotFoundError("Upload the previously prepared capsule to local disk, or provide readable completed benchmark/bundle paths")
            manifest = json.loads(read_once(args.benchmark / "benchmark_manifest.json"))
            members = capsule_members(manifest, read_once(args.bundle), lambda name: read_once(args.benchmark/name), config)
            common.immutable(args.local_archive, stored_zip(members))
            publish = args.archive is not None
    else:
        print("Reusing local input capsule; no Drive input reads", flush=True)
    members, receipt = inspect_capsule(args.local_archive, config)
    # All hashes/inventory are checked before any member is materialized.
    for name, payload in members.items(): common.immutable(args.inputs / name, payload)
    common.immutable(args.inputs / "local_input_receipt.json", common.encoded(receipt))
    if publish:
        try:
            copy_once(args.local_archive, args.archive, receipt["sha256"])
            common.immutable(args.archive.with_suffix(".json"), common.encoded(receipt))
            print("Reusable input capsule saved to Drive once: " + str(args.archive), flush=True)
        except OSError as error:
            print(f"WARNING: could not save the input capsule to Drive ({error}). Local verified inputs are ready. Download {args.local_archive} before deleting this runtime.", flush=True)
    print(json.dumps({"complete": True, **receipt, "local_inputs": str(args.inputs), "training_performed": False}, indent=2), flush=True)


def validate_registration(registration, config):
    if (registration["config"] != config or registration["benchmark_manifest_sha256"] != config["benchmark_manifest_sha256"]
            or registration["model_registration"]["artifact_sha256"] != config["checkpoint_sha256"]
            or registration["training_performed"] or registration["heldout_scoring_authorized"]
            or registration["producer_sha256"]["scripts/evaluate_parking_v0_dev.py"] != EVALUATOR_SHA):
        raise ValueError("Progress archive belongs to another V0, benchmark or evaluator")


def checkpoint_bytes(output, manifest, config, complete=False, verified_prefix=None):
    registration_raw = read_once(output / "v0_registration.json")
    registration = json.loads(registration_raw); validate_registration(registration, config)
    run_id = common.identity(registration)
    members = {"v0_registration.json": registration_raw}; completed = 0
    inventory = v0.dev_inventory(manifest, config)
    if verified_prefix is not None:
        if not 1 <= verified_prefix <= len(inventory): raise ValueError("Invalid verified checkpoint prefix")
        inventory = inventory[:verified_prefix]
    for shard, rows in inventory:
        relative = "predictions/"+Path(shard["file"]).stem+".zip"
        path = output / relative
        if not path.exists(): continue
        payload = read_once(path)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            value = json.loads(archive.read("prediction_manifest.json"))
            if value["run_identity"] != run_id or [s["camera_token"] for s in value["samples"]] != [s["camera_token"] for s in rows]:
                raise ValueError("Local prediction checkpoint identity/order differs")
        members[relative] = payload; completed += 1
    if not completed or (verified_prefix is not None and completed != verified_prefix):
        raise ValueError("A verified checkpoint prefix must contain every completed shard in that prefix")
    if complete:
        if completed != len(v0.dev_inventory(manifest, config)): raise ValueError("Final checkpoint needs the full dev inventory")
        report = json.loads(read_once(output / "v0_dev_results.json"))
        if not report["complete"] or report["run_identity"] != run_id or report["heldout_images_opened"] != 0:
            raise ValueError("No complete development-only result")
        for name in ["v0_dev_results.json", "parking_v0_dev_results.zip"]: members[name] = read_once(output/name)
    members["checkpoint_inventory.json"] = common.encoded({"revision": REVISION, "run_identity": run_id,
        "benchmark_manifest_sha256": config["benchmark_manifest_sha256"], "complete": complete,
        "prediction_shards": completed, "files": {name: common.sha(payload) for name, payload in members.items()}})
    return stored_zip(members), completed


def restore_checkpoint(path, output, manifest, config):
    _, members = read_archive(path)
    index = json.loads(members["checkpoint_inventory.json"])
    registration = json.loads(members["v0_registration.json"]); validate_registration(registration, config)
    allowed = {"v0_registration.json", "v0_dev_results.json", "parking_v0_dev_results.zip"}
    allowed |= {"predictions/"+Path(s["file"]).stem+".zip" for s, _ in v0.dev_inventory(manifest, config)}
    if (set(members) != set(index["files"]) | {"checkpoint_inventory.json"} or not set(index["files"]) <= allowed
            or index["run_identity"] != common.identity(registration)
            or index["benchmark_manifest_sha256"] != config["benchmark_manifest_sha256"]
            or index["prediction_shards"] != sum(name.startswith("predictions/") for name in index["files"])):
        raise ValueError("Progress archive inventory or run identity differs")
    for name, digest in index["files"].items():
        if common.sha(members[name]) != digest: raise ValueError("Progress archive bytes changed")
    # Restore only state used by resume. Metrics are recomputed by the unchanged
    # evaluator, which also revalidates every cached vector against frozen GT.
    for name in index["files"]:
        if name == "v0_registration.json" or name.startswith("predictions/"):
            common.immutable(output/name, members[name])
    return index["prediction_shards"]


def latest_checkpoint(folder):
    for name in ["v0_complete.zip"] + [f"v0_progress_{n:03d}.zip" for n in [20, 16, 12, 8, 4]]:
        path = folder/name
        if path.exists(): return path
    return None


def run_local(args):
    config = v0.configuration()
    if common.file_sha(Path(v0.__file__)) != EVALUATOR_SHA: raise ValueError("The reviewed numerical evaluator changed")
    for path in [args.inputs, args.output, args.local, args.source, args.weights]:
        if str(path.resolve()).startswith("/content/drive/"):
            raise ValueError("All evaluation inputs, outputs, source, weights and scratch must be local /content paths")
    manifest = json.loads(read_once(args.inputs / "benchmark/benchmark_manifest.json"))
    v0.dev_inventory(manifest, config)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoints is not None and not (args.output / "v0_registration.json").exists():
        saved = latest_checkpoint(args.checkpoints)
        if saved is not None:
            print(f"Restoring one progress archive: {saved}", flush=True)
            restore_checkpoint(saved, args.output, manifest, config)
    command = [sys.executable, "-X", "faulthandler", "-u", str(Path(v0.__file__)),
        "--benchmark", str(args.inputs / "benchmark"), "--bundle", str(args.inputs / "model/parking_shared_adapter_results.zip"),
        "--output", str(args.output), "--local", str(args.local), "--source", str(args.source), "--weights", str(args.weights), "--device", "cpu"]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log = args.output/"colab_logs"/("v0_local_"+stamp+".log"); log.parent.mkdir(parents=True, exist_ok=True)
    print("Local evaluation log: " + str(log), flush=True)
    progress = 0; last_saved = 0; drive_available = True

    def save(complete=False):
        nonlocal drive_available, last_saved
        raw, count = checkpoint_bytes(args.output, manifest, config, complete,
            verified_prefix=None if complete else progress)
        name = "v0_complete.zip" if complete else f"v0_progress_{count:03d}.zip"
        local_archive = args.output/"portable_progress"/name
        common.immutable(local_archive, raw); last_saved = count
        if args.checkpoints is not None and drive_available:
            try:
                copy_once(local_archive, args.checkpoints/name, common.sha(raw))
                print("Saved one durable progress archive: " + name, flush=True)
            except OSError as error:
                drive_available = False
                print(f"WARNING: Drive checkpoint save failed ({error}). Further Drive saves are disabled for this invocation. Evaluation continues locally; download {local_archive} before closing the runtime.", flush=True)

    # A Colab cell interruption terminates this wrapper. Forward termination to
    # the evaluator so an orphan cannot keep writing while the user resumes.
    previous_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame): raise KeyboardInterrupt("Local evaluation wrapper was stopped")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True); stream.write(line); stream.flush()
                    if re.match(r"^V0 dev cached/verified \d+/\d+ images ", line):
                        progress += 1
                        if progress % args.checkpoint_every == 0: save()
                code = process.wait()
            except BaseException:
                if process.poll() is None: process.terminate()
                process.wait()
                raise
            finally:
                process.stdout.close()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if code:
        if (args.output/"v0_registration.json").exists() and progress > last_saved:
            # Preserve a local emergency snapshot; durable progress still uses
            # the known periodic names and never scans large Drive folders.
            raw, count = checkpoint_bytes(args.output, manifest, config, verified_prefix=progress)
            common.immutable(args.output/"portable_progress"/f"v0_interrupted_{count:03d}.zip", raw)
        raise RuntimeError(f"V0 exited {code}; inputs and predictions are preserved locally; inspect {log}")
    save(complete=True)
    print("Return local result ZIP: " + str(args.output/"parking_v0_dev_results.zip"), flush=True)
    if not drive_available: print("IMPORTANT: no final Drive save succeeded after the warning. Download the result ZIP and v0_complete.zip now.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    p = commands.add_parser("stage-inputs")
    for name in ["local-archive", "inputs"]: p.add_argument("--"+name, type=Path, required=True)
    for name in ["archive", "benchmark", "bundle"]: p.add_argument("--"+name, type=Path)
    p = commands.add_parser("evaluate-local")
    for name in ["inputs", "output", "local", "source", "weights"]: p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--checkpoints", type=Path)
    p.add_argument("--checkpoint-every", type=int, choices=[4], default=4)
    args = parser.parse_args()
    stage(args) if args.stage == "stage-inputs" else run_local(args)


if __name__ == "__main__": main()
