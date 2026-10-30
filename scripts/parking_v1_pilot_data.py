"""Archive-first inputs for the reviewed V1 pilot; no model imports or training.

Only train bodies and the existing dev-only capsule are materialized. The full
benchmark's held-out metadata is used solely to enforce split separation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import parking_v0_local_io as local
from scripts import prepare_parking_v1_training as prepare

CONFIG = ROOT / "configs/parking_v1_matched_pilot_v1.json"


def configuration():
    c = json.loads(CONFIG.read_text())
    proposal = ROOT / "configs/parking_v1_near_field_proposal_v1.json"
    if (common.file_sha(proposal) != c["proposal_sha256"] or c["batch_size"] != 8
            or c["blocks"] != 5 or c["updates_per_block"] != 384 or c["updates_per_arm"] != 1920
            or c["blocks"] * c["updates_per_block"] != c["updates_per_arm"]
            or c["heldout_evaluation_authorized"] or c["phone_update_authorized"]
            or c["automatic_extension_authorized"] or c["accuracy_threshold_m"] is not None):
        raise ValueError("The reviewed bounded V1 pilot contract changed")
    declared = json.loads(proposal.read_text())["matched_pilot"]
    for key in ["optimizer", "learning_rate", "weight_decay", "huber_delta_m", "gradient_clip_norm", "updates_per_block", "updates_per_arm"]:
        if c[key] != declared[key]: raise ValueError("Pilot differs from the prospective recipe: " + key)
    if (c["arms"] != list(declared["arms"]) or c["blocks"] != declared["optimization_blocks_per_arm"]
            or c["train_images"] != 4096 or c["dev_images"] != 512 or c["trainable_parameters"] != 1553
            or c["adapter_width"] != 16 or (c["near_loss_weight"], c["broad_loss_weight"]) != (.8, .2)):
        raise ValueError("Pilot architecture, data or loss budget differs from the reviewed proposal")
    return c


def read_summary(path, c):
    if common.file_sha(path) != c["preparation_summary_sha256"]:
        raise ValueError("Use the reviewed completed V1 preparation summary, unchanged")
    names = {"preparation_identity.json", "preparation_report.json", "selection_manifest.json", "training_input_manifest.json"}
    with zipfile.ZipFile(path) as z:
        if (set(z.namelist()) != names or len(z.namelist()) != 4
                or sum(i.file_size for i in z.infolist()) > 100_000_000 or z.testzip() is not None):
            raise ValueError("Unexpected preparation summary inventory/CRC")
        values = {n: prepare.coverage.check_signed(json.loads(z.read(n))) for n in names}
    identity, selection, manifest, report = [values[n] for n in
        ["preparation_identity.json", "selection_manifest.json", "training_input_manifest.json", "preparation_report.json"]]
    pc = prepare.configuration(); protocol = prepare.data_protocol(pc)
    roles = prepare.data.verified_roles(ROOT / pc["frozen_roles_file"], 850)
    if (identity["manifest_sha256"] != c["preparation_identity_sha256"]
            or manifest["manifest_sha256"] != c["training_manifest_sha256"]
            or manifest["config"] != pc or identity["config"] != pc
            or identity["data_protocol"] != protocol or manifest["near_mask_protocol"] != protocol
            or manifest["preparation_identity_sha256"] != identity["manifest_sha256"]
            or manifest["selection_sha256"] != selection["manifest_sha256"]
            or report["training_input_manifest_sha256"] != manifest["manifest_sha256"]
            or not report["complete"] or not manifest["complete"] or not manifest["training_data_ready"]
            or not report["training_data_ready"] or manifest["dev_sensor_samples_exported"]
            or manifest["heldout_sensor_samples_exported"]):
        raise ValueError("Preparation identity or training readiness differs")
    for name, digest in identity["producer_sha256"].items():
        local.v0.safe_name(name)
        if common.file_sha(ROOT / name) != digest: raise ValueError("Frozen preparation source changed: " + name)
    prepare.validate_selection(selection, identity, pc, roles, protocol)
    samples = manifest["samples"]
    if (len(samples) != c["train_images"] or len({s["camera_token"] for s in samples}) != len(samples)
            or len({s["recording_log"] for s in samples}) != c["train_recording_logs"]
            or manifest["selected_pool"] != selection["selected_pool"]):
        raise ValueError("Reviewed training support changed")
    return identity, selection, manifest


def disjoint(train, benchmark):
    for key in ["camera_token", "sample_token", "recording_log"]:
        if {s[key] for s in train}.intersection(s[key] for s in benchmark["samples"]):
            raise ValueError("Train/benchmark overlap: " + key)
    if any(s["role"] != "train" for s in train): raise ValueError("Training bodies must be train only")


def local_path(path):
    path = Path(path)
    if path.is_symlink() or str(path.resolve()).startswith("/content/drive/"):
        raise ValueError("Keep working inputs, features, logs and checkpoints on local disk")
    return path


def paths_separate(paths):
    paths = [Path(p).resolve() for p in paths]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError("Keep input, feature-cache, output, source and weight folders separate")


def transfer(source, target, digest, maximum=250_000_000):
    """Reuse a verified local archive without touching its Drive source."""
    if target.exists():
        if common.file_sha(target) != digest: raise ValueError("Preserve conflicting local archive")
        return
    try:
        prepare.atomic_copy(source, target, digest, maximum)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required existing archive is missing: {source}. Upload that ZIP to /content "
            "or correct its source path in section 1. Do not rerun dataset preparation or training to recreate it.") from error
    except OSError as error:
        if error.errno == 5:
            raise RuntimeError("Drive input/output failure is possible, not confirmed quota. No retry/remount. "
                "Preserve the run; upload the same archive to /content or wait for Drive recovery.") from error
        raise


def stage(a):
    c = configuration(); local_path(a.inputs)
    paths_separate([a.inputs, a.summary, a.dev_capsule, a.v0_result, a.parts])
    a.inputs.mkdir(parents=True, exist_ok=True)
    owner = a.inputs / "owner.json"
    if not owner.exists() and any(a.inputs.iterdir()): raise ValueError("Use an empty, owned local input directory")
    common.immutable(owner, common.encoded({"revision": c["revision"], "config_sha256": common.identity(c)}))
    if shutil.disk_usage(a.inputs).free < 2 * 1024**3:
        raise ValueError("Need at least 2 GiB free for local input staging")
    archives = a.inputs / "archives"
    for source, name, digest in [(a.summary, "summary.zip", c["preparation_summary_sha256"]),
            (a.dev_capsule, "dev_inputs.zip", c["v0_input_capsule_sha256"]),
            (a.v0_result, "v0_complete.zip", c["v0_result_sha256"])]:
        transfer(source, archives / name, digest)
    identity, selection, manifest = read_summary(archives / "summary.zip", c)
    members, _ = local.inspect_capsule(archives / "dev_inputs.zip", local.v0.configuration())
    benchmark = json.loads(members["benchmark/benchmark_manifest.json"])
    disjoint(manifest["samples"], benchmark)
    for name, payload in members.items(): common.immutable(a.inputs / "dev" / name, payload)
    # The existing result has no sensor bodies and is pinned byte-for-byte.
    from scripts.diagnose_parking_v0_errors import verify_result
    verify_result(archives / "v0_complete.zip", local.v0.configuration())
    for part in manifest["parts"]:
        local.v0.safe_name(part["file"])
        target = archives / Path(part["file"]).name
        transfer(a.parts / Path(part["file"]).name, target, part["sha256"])
        if target.stat().st_size != part["size_bytes"]: raise ValueError("Training archive size differs")
        number = int(target.stem[-2:])
        checked = prepare.inspect_part(target, selection, number, prepare.configuration(), identity, manifest["near_mask_protocol"])
        with zipfile.ZipFile(target) as z:
            for shard in checked["shards"]:
                common.immutable(a.inputs / "train" / shard["file"], z.read(shard["file"]))
        print(f"Staged/verified train part {number:02d}: {part['images']} images (local reads only after transfer)", flush=True)
    common.immutable(a.inputs / "training_input_manifest.json", common.encoded(manifest))
    receipt = {"complete": True, "revision": c["revision"], "config_sha256": common.identity(c),
        "training_manifest_sha256": manifest["manifest_sha256"], "train_images": len(manifest["samples"]),
        "dev_images": c["dev_images"], "benchmark_manifest_sha256": benchmark["manifest_sha256"],
        "raw_train_part_bodies_verified": True, "heldout_bodies_opened": 0,
        "train_benchmark_camera_keyframe_log_overlap": 0, "training_performed": False,
        "warnings": ["No CAM_FRONT Vehicle 0-2m training support.", "0-1m remains exploratory.",
            "Camera-axis Z is not ultrasonic nearest-surface distance; no 10cm qualification."]}
    common.immutable(a.inputs / "prepared_inputs.json", common.encoded(receipt))
    print(json.dumps(receipt, indent=2), flush=True)


def load_prepared(inputs):
    c = configuration(); local_path(inputs)
    receipt = json.loads(local.read_once(inputs / "prepared_inputs.json"))
    if not receipt["complete"] or receipt["config_sha256"] != common.identity(c):
        raise ValueError("Run CPU staging first with the same contract")
    manifest = prepare.coverage.check_signed(json.loads(local.read_once(inputs / "training_input_manifest.json")))
    if manifest["manifest_sha256"] != c["training_manifest_sha256"]: raise ValueError("Training manifest changed")
    benchmark = json.loads(local.read_once(inputs / "dev/benchmark/benchmark_manifest.json"))
    inventory = local.v0.dev_inventory(benchmark, local.v0.configuration())
    disjoint(manifest["samples"], benchmark)
    return c, receipt, manifest, benchmark, inventory


def training_shards(manifest):
    grouped = defaultdict(list)
    for sample in manifest["samples"]:
        if sample["role"] != "train": raise ValueError("Forbidden training role")
        local.v0.safe_name(sample["shard_member"])
        if not sample["shard_member"].startswith("shards/train/"): raise ValueError("Forbidden training shard")
        grouped[sample["shard_member"]].append(sample)
    return sorted(grouped.items())


def load_train_shard(inputs, name, rows, protocol):
    import numpy as np
    from PIL import Image
    raw = local.read_once(inputs / "train" / name)
    if len({s["shard_sha256"] for s in rows}) != 1 or common.sha(raw) != rows[0]["shard_sha256"]:
        raise ValueError("Local training shard hash differs")
    result = {}; expected = {"shard_manifest.json"}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        value = prepare.coverage.check_signed(json.loads(z.read("shard_manifest.json")))
        metadata = [{k: v for k, v in s.items() if k not in ["part_archive", "shard_member", "shard_sha256"]} for s in rows]
        if value["samples"] != metadata: raise ValueError("Local training shard metadata differs")
        for sample in rows:
            if sample["role"] != "train": raise ValueError("No dev/heldout training bodies")
            record = {}
            for kind in ["image", "ground_truth", "broad"]:
                member = sample[kind+"_member"]; local.v0.safe_name(member); expected.add(member)
                payload = z.read(member)
                if common.sha(payload) != sample[kind+"_sha256"]: raise ValueError("Training member changed")
                if kind == "image":
                    with Image.open(io.BytesIO(payload)) as image:
                        if image.mode != "RGB" or image.size != (sample["width"], sample["height"]):
                            raise ValueError("Training RGB contract changed")
                        record["rgb"] = np.asarray(image).copy()
                else:
                    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                        record[kind] = {k: arrays[k] for k in arrays.files}
            prepare.metrics.validate_gt(record["ground_truth"], sample, protocol)
            prepare.validate_broad(record["broad"], sample)
            if len(record["broad"]["z"]) != sample["broad_reference_points"]: raise ValueError("Broad count changed")
            result[sample["camera_token"]] = record
        if set(z.namelist()) != expected or len(z.namelist()) != len(expected) or z.testzip() is not None:
            raise ValueError("Local training shard inventory/CRC differs")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["inputs", "summary", "parts", "dev-capsule", "v0-result"]:
        p.add_argument("--"+key, type=Path, required=True)
    stage(p.parse_args())


if __name__ == "__main__": main()
