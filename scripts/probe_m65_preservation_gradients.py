"""M65b prospective train-only gradient measurement, with zero optimizer steps."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys

# Absolute script launches expose scripts/, not the repository's third_party/.
# Resolve imports from this file, independently of the notebook's working dir.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import m65_a2_preservation_control as c
import m64_teacher_qualification as q
from diagnose_m65_preservation_regression import REVIEWED_MANIFEST

REVISION = "M65B-ZERO-UPDATE-GRADIENTS-2026-10-05-r2"
CONTROL_SHA = "e5c2b592977c68f79bc371196536c0263badaa7277e7fc17bd0ed6c7b64a6a18"
PARTS = ("loss_ce", "loss_bbox", "loss_giou", "loss_depth", "loss_dim",
         "loss_angle", "loss_center", "loss_depth_map")


def parameter_group(name):
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("depthaware_transformer.encoder."):
        return "visual_encoder"
    if name.startswith("depthaware_transformer.decoder."):
        return "depth_aware_decoder"
    if name.startswith("depth_predictor."):
        return "dense_depth_predictor"
    if name.startswith("depth_embed."):
        return "object_depth_head"
    if name.startswith("angle_embed."):
        return "angle_head"
    if name.startswith("class_embed."):
        return "class_head"
    if name.startswith("input_proj."):
        return "input_projection"
    return "other"


def gradient_stats(names, gt, keep):
    """Aggregate tensor dot products, not means of per-tensor cosines."""
    sums = {}
    for name, a, b in zip(names, gt, keep):
        for group in ("all", parameter_group(name)):
            value = sums.setdefault(group, dict(gt_sq=0., keep_sq=0., dot=0.))
            if a is not None:
                value["gt_sq"] += float(a.double().square().sum())
            if b is not None:
                value["keep_sq"] += float(b.double().square().sum())
            if a is not None and b is not None:
                value["dot"] += float((a.double() * b.double()).sum())
    result = {}
    for group, value in sums.items():
        a, b = math.sqrt(value["gt_sq"]), math.sqrt(value["keep_sq"])
        result[group] = dict(gt_norm=a, preservation_norm=b,
            preservation_to_gt_norm_ratio=b / a if a > 0 else None,
            cosine=value["dot"] / (a * b) if a > 0 and b > 0 else None)
    return result


def cpu_gradients(loss, parameters):
    import torch
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
        raise RuntimeError("Non-finite diagnostic gradients")
    return [g.detach().cpu() if g is not None else None for g in grads]


def verify_inputs(args):
    """New diagnostic identity: preserve historical manifests and build receipts."""
    from third_party.monodetr.m65_preservation import RULES, WEIGHTS
    m = q.read_json(args.manifest)
    if (m.get("manifest_sha256") != REVIEWED_MANIFEST
            or q.signature({k: v for k, v in m.items() if k != "manifest_sha256"}) != REVIEWED_MANIFEST):
        raise RuntimeError("Use the reviewed, unchanged M65 manifest")
    if (m["recipe"] != c.RECIPE or m["implementation_sha256"] != c.implementation_hash()
            or m["preservation_rules"] != RULES or m["preservation_weights"] != WEIGHTS
            or m["kd_authorized"] is not False):
        raise RuntimeError("M65 recipe or frozen implementation changed")
    entry = m["models"]["a2"]
    if q.sha(entry["checkpoint"]) != q.A2_SHA or q.sha(c.checkpoint_path(m)) != CONTROL_SHA:
        raise RuntimeError("Original A2 or completed M65 checkpoint differs")
    receipt = q.read_json(c.checkpoint_path(m).with_suffix(".json"))
    if (receipt.get("complete") is not True or receipt.get("checkpoint_sha256") != CONTROL_SHA
            or receipt.get("manifest_sha256") != REVIEWED_MANIFEST):
        raise RuntimeError("Invalid completed M65 checkpoint receipt")
    if q.source_hash(args.repo) != entry["patched_source_sha256"]:
        raise RuntimeError("Fresh patched source does not match reviewed A2 source")
    build_path = args.repo / "lib/models/monodetr/ops/m64_build_receipt.json"
    build = q.read_json(build_path)
    if not build.get("import_verified") or q.sha(build["binary"]) != build["binary_sha256"]:
        raise RuntimeError("Fresh local attention build is missing or changed")
    if c.data_identity(args.dataset_root) != m["data_identity"]:
        raise RuntimeError("KITTI splits, labels or calibration differ")
    env = q.environment()
    if env["gpu"] is None or env["cuda"] != "13.0":
        raise RuntimeError("Use the tested private CUDA13 runtime and a GPU")
    view = copy.deepcopy(m)
    view["models"]["a2"].update(repo=str(args.repo), attention_build_receipt=str(build_path),
        attention_binary=build["binary"], attention_binary_sha256=build["binary_sha256"])
    view["dataset_root"] = str(args.dataset_root)
    view["training_config"]["dataset"]["root_dir"] = str(args.dataset_root)
    identity = dict(revision=REVISION, historical_manifest_sha256=REVIEWED_MANIFEST,
        original_checkpoint_sha256=q.A2_SHA, control_checkpoint_sha256=CONTROL_SHA,
        implementation_sha256=c.implementation_hash(), probe_script_sha256=q.sha(Path(__file__)),
        environment=env, original_environment=m["environment"],
        historical_runtime_equivalence_claimed=False, role=args.role,
        build_receipt_sha256=q.sha(build_path), source_sha256=q.source_hash(args.repo),
        dataset_identity=m["data_identity"], seed=444, batches=8, batch_size=4,
        image_selection="First32 train IDs in saved train order, not selected using validation errors",
        augmentation="Same M65 online augmentation settings, seeded prospectively")
    return m, view, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("original", "control"))
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    original, view, identity = verify_inputs(args)
    if args.report.exists():
        existing = q.read_json(args.report)
        if existing.get("complete") and existing.get("identity") == identity:
            print(f"Complete zero-update probe already exists: {args.report}", flush=True)
            return
        raise RuntimeError("Preserve the previous report and choose a fresh diagnostic RUN_ID")
    import torch
    from third_party.monodetr.m65_preservation import WEIGHTS
    c.seed_all(444)
    model, anchor, criterion, dataset = c.training_runtime(view)
    if args.role == "control":
        payload = q.safe_payload(c.checkpoint_path(original))
        if payload.get("m65_manifest_sha256") != REVIEWED_MANIFEST or payload.get("epoch") != 1:
            raise RuntimeError("Control checkpoint lineage mismatch")
        c.validate_training_summary(original, payload.get("training_summary", {}))
        model.load_state_dict(payload["model_state"], strict=True)
        del payload
    # Neither role creates an optimizer. All parameters and buffers must remain unchanged.
    model_before, anchor_before = q.tensor_hash(model.state_dict()), q.tensor_hash(anchor.state_dict())
    names, parameters = zip(*[(name, p) for name, p in model.named_parameters() if p.requires_grad])
    subset = torch.utils.data.Subset(dataset, range(32))
    loader = torch.utils.data.DataLoader(subset, batch_size=4, shuffle=False, num_workers=0)
    c.seed_all(444)
    rows = []
    for index, batch in enumerate(loader, 1):
        print(f"{REVISION} {args.role}: gradients batch {index}/8 (optimizer steps=0)", flush=True)
        total, gt, keep, counts, parts, outputs, reference, targets = c.losses(model, anchor, criterion, batch)
        a = cpu_gradients(gt, parameters)
        b = cpu_gradients(keep["total"] * c.m_preservation_weight(), parameters)
        statistics = gradient_stats(names, a, b)
        del b
        terms = {}
        for prefix in PARTS:
            selected = [value * criterion.weight_dict[key] for key, value in parts.items()
                        if key in criterion.weight_dict and (key == prefix
                        or (key.startswith(prefix + "_") and key[len(prefix) + 1:].isdigit()))]
            if selected:
                value = sum(selected)
                grads = cpu_gradients(value, parameters)
                terms[prefix] = dict(weighted_loss=float(value.detach()),
                                    gradients=gradient_stats(names, a, grads))
                del grads
        components = {}
        for name, weight in WEIGHTS.items():
            grads = cpu_gradients(keep[name] * weight, parameters)
            components[name] = dict(weighted_loss=float(keep[name].detach()) * weight,
                                   gradients=gradient_stats(names, a, grads))
            del grads
        del a
        uncertainty = outputs["pred_depth"][..., 1] - reference["pred_depth"][..., 1]
        rows.append(dict(batch=index, sample_ids=[f"{int(i):06d}" for i in batch[3]["img_id"]],
            gt_loss=float(gt.detach()), preservation_loss=float(keep["total"].detach()),
            paired_counts=counts, gt_vs_preservation=statistics, gt_components=terms,
            preservation_components=components,
            all_query_log_uncertainty_delta_mean=float(uncertainty.detach().mean()),
            all_query_log_uncertainty_abs_delta_mean=float(uncertainty.detach().abs().mean())))
        del total, gt, keep, parts, outputs, reference, targets, uncertainty
    unchanged = model_before == q.tensor_hash(model.state_dict())
    anchor_unchanged = anchor_before == q.tensor_hash(anchor.state_dict())
    if not unchanged or not anchor_unchanged or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Probe changed state or populated parameter.grad")
    report = dict(complete=True, identity=identity, rows=rows, optimizer_steps=0,
        model_parameters_and_buffers_unchanged=unchanged, anchor_unchanged=anchor_unchanged,
        checkpoint_files_unchanged=(q.sha(c.checkpoint_path(original)) == CONTROL_SHA
                                    and q.sha(original["models"]["a2"]["checkpoint"]) == q.A2_SHA),
        interpretation=["Norms are un-clipped loss gradients; negative cosine indicates local opposing directions.",
            "For gt_components/preservation_components, gt_norm is the total GT norm and preservation_norm is the named component norm.",
            "Original-A2 preservation gradients are expected to be near zero at equality; do not treat that alone as a failure.",
            "This is a prospective endpoint probe, not a replay or causal attribution of the 928 historical AdamW updates.",
            "All-query uncertainty differences are not object-matched ranking changes or an AP decomposition."],
        training_authorized=False, external_kd_enabled=False, checkpoint_promotion_authorized=False)
    if not report["checkpoint_files_unchanged"]:
        raise RuntimeError("Checkpoint files changed during the probe")
    q.write_json(args.report, report)
    print(f"Complete: {args.report}. Zero optimizer steps; stop for review.", flush=True)


if __name__ == "__main__":
    main()
