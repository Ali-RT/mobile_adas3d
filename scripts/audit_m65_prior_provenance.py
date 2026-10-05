"""Audit released MonoPRIO prior provenance without authorizing KD."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess

from m64_teacher_qualification import ASSET_SHA256, TEACHER_COMMIT, sha, write_json

BUILDER_SHA = "bae797cf75febd65c17b4f566ce3aaebd7680f562e0dbb9162d04cf2203e5077"
CONFIG_SHA = "bb9be0955669624b990aca270620a8c46c2ef0eaa951fee9cfc1a3e8ab1a5006"


def default_maxima(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "default_class_cfgs"]
    if len(functions) != 1:
        raise RuntimeError("Unreviewed prior configuration layout")
    returned = [n for n in functions[0].body if isinstance(n, ast.Return)]
    if len(returned) != 1 or not isinstance(returned[0].value, ast.Dict):
        raise RuntimeError("Expected literal per-class prior recipe")
    result = {}
    for name, call in zip(returned[0].value.keys, returned[0].value.values):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "ClassPriorCfg":
            raise RuntimeError("Unexpected class prior recipe")
        params = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        result[ast.literal_eval(name)] = params["k_geo"] * params["k_vis"]
    if set(result) != {"Car", "Pedestrian", "Cyclist"}:
        raise RuntimeError("Unexpected prior taxonomy")
    return result


def inspect(bank, maxima):
    import numpy as np
    names = bank["class_names"].tolist()
    counts = bank["class_counts"].tolist()
    offsets = bank["class_offsets"].tolist()
    if names != ["Pedestrian", "Car", "Cyclist"] or len(counts) != 3:
        raise RuntimeError("Unexpected released bank class order")
    if offsets != [0, counts[0], sum(counts[:2]), sum(counts)]:
        raise RuntimeError("Inconsistent prior offsets/counts")
    if any(not isinstance(count, int) or count <= 0 for count in counts):
        raise RuntimeError("Invalid prior class counts")
    if any(bank[key].shape[0] != sum(counts) for key in ("visual", "mu", "sigma", "mu_log", "V_log", "inv_std_log")):
        raise RuntimeError("Inconsistent prototype tensor lengths")
    if any(not np.isfinite(bank[key]).all() for key in ("visual", "mu", "sigma", "mu_log", "V_log", "inv_std_log")):
        raise RuntimeError("Non-finite prior tensors")
    metadata = {"sample_ids", "train_ids", "split_sha256", "label_tree_sha256", "construction_receipt"}
    rows = {name: dict(released_prototypes=int(count), default_maximum=maxima[name],
                       compatible_with_default_recipe=int(count) <= maxima[name])
            for name, count in zip(names, counts)}
    incompatible = [name for name, row in rows.items() if not row["compatible_with_default_recipe"]]
    conclusion = "The bank's construction inputs are not independently verified. "
    if incompatible:
        conclusion += "/".join(incompatible) + " counts exceed the pinned default recipe. "
    conclusion += "Obtain the exact release construction receipt or reproduce the bank before external-teacher KD."
    return dict(complete=True, classes=rows, bank_keys=sorted(bank.keys()),
                construction_identifiers_present=bool(metadata & set(bank)),
                default_recipe_compatible=all(v["compatible_with_default_recipe"] for v in rows.values()),
                prior_training_ids_independently_verified=False,
                validation_label_leakage_proven=False, kd_authorized=False,
                conclusion=conclusion)


def main():
    import numpy as np
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-repo", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    repo = a.teacher_repo.resolve()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != TEACHER_COMMIT:
        raise RuntimeError("Wrong teacher source pin")
    builder, config = repo / "tools/build_priors.py", repo / "configs/monoprio.yaml"
    if sha(builder) != BUILDER_SHA or sha(config) != CONFIG_SHA or sha(a.prior) != ASSET_SHA256["prior"]:
        raise RuntimeError("Prior/builder/config bytes differ from the reviewed release")
    with np.load(a.prior, allow_pickle=False) as values:
        bank = {k: values[k] for k in values.files}
    report = inspect(bank, default_maxima(builder.read_text()))
    report.update(upstream_commit=TEACHER_COMMIT, builder_sha256=BUILDER_SHA,
                  config_sha256=CONFIG_SHA, prior_sha256=sha(a.prior),
                  configured_train_split=yaml.safe_load(config.read_text())["dataset"]["train_split"],
                  teacher_independent_a2_control_allowed=True)
    write_json(a.output, report)
    import json
    print(json.dumps(report, indent=2), flush=True)
    print("External-teacher KD remains disabled. This does not block the original-A2 preservation control.", flush=True)


if __name__ == "__main__":
    main()
