"""Prepare the original A2 source and a fresh M68 CUDA/ONNX execution record."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import sys

import audit_m67_a2_coreml as historical
import m64_teacher_qualification as q
import m68_a2_runtime as runtime
from setup_m64_runtime import runtime_env

ROOT = Path(__file__).resolve().parents[1]
A2_URL = "https://github.com/ZrrSkywalker/MonoDETR.git"


def run(command, *, env=None, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, cwd=cwd, check=True)


def ensure_repo(repo, source):
    if not repo.name.startswith("MonoDETR_M68"):
        raise RuntimeError("Use a dedicated MonoDETR_M68 checkout")
    if not repo.exists():
        run(["git", "clone", "--no-checkout", A2_URL, repo])
        run(["git", "checkout", "--detach", q.A2_COMMIT], cwd=repo)
    if not (repo / ".git").is_dir():
        raise RuntimeError(f"Preserve the existing non-Git directory: {repo}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip().rstrip("/")
    if head != q.A2_COMMIT or origin != A2_URL:
        raise RuntimeError(f"Unexpected source identity at {repo}; preserve it")
    approved = {source["native_source_sha256"], source["export_source_sha256"]}
    tracked_clean = (subprocess.run(["git", "diff", "--quiet"], cwd=repo).returncode == 0
                     and subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode == 0)
    if not tracked_clean and q.source_hash(repo) not in approved:
        raise RuntimeError(f"Unreviewed tracked source edits at {repo}; preserve them")


def freeze(args, source):
    repo, dataset = args.repo.resolve(), args.dataset_root.resolve()
    prepared = args.output_root.resolve() / "prepared"
    receipt_path = prepared / "m68_runtime_receipt.json"
    receipt = runtime.validate_current_runtime(receipt_path)
    if q.source_hash(repo) != source["export_source_sha256"]:
        raise RuntimeError("M68 portable A2 source differs from the reviewed source")
    build = q.read_json(repo / "lib/models/monodetr/ops/m64_build_receipt.json")
    binary = Path(build["binary"])
    if (build.get("import_verified") is not True
            or not binary.resolve().is_relative_to(repo)
            or q.sha(binary) != build["binary_sha256"]):
        raise RuntimeError("M68 attention build is missing or changed")
    ids = historical.restore_inputs(dataset, args.splits.resolve(), [p.resolve() for p in args.candidate])
    inventory = historical.fixture_inventory(dataset, ids)
    if ids != source["sample_ids"] or inventory != source["input_files"]:
        raise RuntimeError("M68 fixed16 input bytes/order differ from the reviewed M67 inputs")
    config = copy.deepcopy(source["config"])
    config["dataset"]["root_dir"] = str(dataset)
    value = dict(schema_version=1, revision=runtime.REVISION, repo=str(repo), config=config,
                 checkpoint=source["checkpoint"], checkpoint_sha256=q.A2_SHA, checkpoint_epoch=130,
                 source_m67_manifest=str(args.m67_manifest.resolve()),
                 source_m67_manifest_sha256=q.sha(args.m67_manifest),
                 source_m67_manifest_signature=source["signature_sha256"],
                 runtime_receipt=str(receipt_path), runtime_receipt_sha256=q.sha(receipt_path),
                 environment=receipt["environment"], implementation_sha256=runtime.implementation_hash(),
                 native_source_sha256=source["native_source_sha256"],
                 export_source_sha256=source["export_source_sha256"],
                 attention_binary=str(binary), attention_binary_sha256=q.sha(binary),
                 dataset_root=str(dataset), sample_ids=ids, input_files=inventory,
                 historical_runtime_reused=False, precision="FP32", optimizer_steps=0,
                 training_authorized=False, deployment_authorized=False)
    value["signature_sha256"] = q.signature(value)
    path = prepared / "m68_a2_manifest.json"
    if path.exists() and q.read_json(path) != value:
        raise RuntimeError("M68 prepared inputs changed; preserve this run and use a new RUN_ID")
    q.write_json(path, value)
    runtime.load_manifest(path)
    print(f"M68 ready: checkpoint, portable source, new CUDA binary and fixed16 inputs verified: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m67-manifest", type=Path, required=True)
    parser.add_argument("--m66-manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--freeze", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source = runtime.historical_inputs(args.m67_manifest.resolve(), args.m66_manifest.resolve())
    if args.freeze:
        freeze(args, source)
        return
    cuda = Path("/usr/local/cuda-13.0")
    if not (cuda / "bin/nvcc").is_file():
        cuda = Path("/usr/local/cuda")
    python = args.venv.resolve() / "bin/python"
    receipt = args.output_root.resolve() / "prepared/m68_runtime_receipt.json"
    run([sys.executable, ROOT / "scripts/setup_m68_onnx_runtime.py",
         "--venv", args.venv, "--cuda", cuda, "--receipt", receipt])
    env = runtime_env(cuda.resolve(), python.parent)
    repo = args.repo.resolve()
    ensure_repo(repo, source)
    for script, flags in (
        ("patch_monodetr_colab_compat.py", ["--monodetr-repo", repo]),
        ("patch_monodetr_product_taxonomy.py", ["--monodetr-repo", repo]),
        ("patch_monodetr_mobilenetv4.py", ["--monodetr-repo", repo]),
        ("patch_m64_inference_sources.py", ["--repo", repo, "--role", "a2"]),
        ("patch_monodetr_coreml_export.py", ["--monodetr-repo", repo]),
    ):
        # The last source patch implements ordinary portable tensor operations;
        # applying it does not import or install a converter.
        run([python, ROOT / "scripts" / script, *flags], env=env)
    if q.source_hash(repo) != source["export_source_sha256"]:
        raise RuntimeError("Reconstructed portable A2 source does not match its reviewed hash")
    run([python, ROOT / "scripts/build_m64_attention.py", "--repo", repo, "--role", "a2"], env=env)
    run([python, "-u", Path(__file__), *sys.argv[1:], "--freeze"], env=env, cwd=ROOT)


if __name__ == "__main__":
    main()
