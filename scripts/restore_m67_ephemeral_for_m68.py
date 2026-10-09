"""Safely reconstruct M67's Colab-only files after a runtime reset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


M67_REVISION = "M67-A2-EXPORT-RUNTIME-AUDIT-2026-10-06-r1"
A2_URL = "https://github.com/ZrrSkywalker/MonoDETR.git"
A2_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
A2_SHA = "ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4"


def run(command: list[str | Path], *, cwd: Path | None = None,
        env: dict[str, str] | None = None) -> None:
    printable = list(map(str, command))
    print("+", " ".join(printable), flush=True)
    subprocess.run(printable, cwd=cwd, env=env, check=True)


def digest(value: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_signed_m67(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    body = {key: value for key, value in manifest.items() if key != "signature_sha256"}
    if digest(body) != manifest.get("signature_sha256"):
        raise RuntimeError("Saved M67 manifest signature is invalid; preserve all artifacts")
    if (manifest.get("revision") != M67_REVISION
            or manifest.get("checkpoint_sha256") != A2_SHA
            or manifest.get("checkpoint_epoch") != 130):
        raise RuntimeError("Saved M67 manifest is not the reviewed A2 epoch-130 run")
    return manifest


def cuda13_path(saved_cuda_home: str) -> Path:
    cuda = Path(saved_cuda_home)
    nvcc = cuda / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(
            f"The saved M67 CUDA toolkit path is unavailable: {nvcc}. "
            "Select a Colab GPU runtime with CUDA 13.0; do not substitute CUDA 12.8/13.1."
        )
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "release 13.0," not in version:
        raise RuntimeError(f"M67 requires its recorded CUDA 13.0 toolkit; found:\n{version}")
    return cuda.resolve()


def repo_state(repo: Path) -> str:
    if not repo.exists():
        run(["git", "clone", "--no-checkout", A2_URL, repo])
        run(["git", "checkout", "--detach", A2_COMMIT], cwd=repo)
    if not (repo / ".git").is_dir():
        raise RuntimeError(f"{repo} exists but is not the expected Git checkout; preserve it")
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip().rstrip("/")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if origin != A2_URL or head != A2_COMMIT:
        raise RuntimeError(f"M67 source checkout identity differs at {repo}; preserve it")
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobile-repo", type=Path, required=True)
    parser.add_argument("--m66-manifest", type=Path, required=True)
    parser.add_argument("--m67-output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    args = parser.parse_args()

    mobile = args.mobile_repo.resolve()
    output = args.m67_output.resolve()
    manifest_path = output / "m67_manifest.json"
    receipt = output / "m67_runtime_receipt.json"
    repo, venv = args.repo.resolve(), args.venv.resolve()
    python = venv / "bin/python"
    m67 = load_signed_m67(manifest_path)
    expected_paths = {
        "output": output,
        "repo": repo,
        "runtime_receipt": receipt,
        "dataset_root": Path("/content/kitti_m67"),
    }
    for key, expected in expected_paths.items():
        if Path(m67[key]).resolve() != expected.resolve():
            raise RuntimeError(f"M67 manifest {key} path differs: {m67[key]} != {expected}")
    if not args.m66_manifest.is_file() or not args.splits.joinpath("val.txt").is_file():
        raise FileNotFoundError("M66 manifest or Chen split file is missing from Drive")
    if not receipt.is_file():
        raise FileNotFoundError(f"M67 runtime receipt is missing on Drive: {receipt}")
    if file_digest(receipt) != m67.get("runtime_receipt_sha256"):
        raise RuntimeError("Saved M67 runtime receipt bytes differ from the signed manifest; preserve artifacts")
    receipt_data = json.loads(receipt.read_text())
    receipt_body = {key: value for key, value in receipt_data.items() if key != "signature_sha256"}
    if (receipt_data.get("complete") is not True
            or receipt_data.get("revision") != M67_REVISION
            or receipt_data.get("coremltools") != "9.0"
            or digest(receipt_body) != receipt_data.get("signature_sha256")
            or Path(receipt_data.get("python", "")).resolve() != python.resolve()):
        raise RuntimeError("Saved M67 runtime receipt is invalid or belongs to another environment")
    if file_digest(args.m66_manifest) != m67.get("original_manifest_file_sha256"):
        raise RuntimeError("M66 input manifest differs from the exact file frozen by M67")

    saved_cuda_home = m67["environment"].get("cuda_home")
    if not saved_cuda_home:
        raise RuntimeError("M67 manifest has no recorded CUDA_HOME; do not guess a replacement")
    cuda = cuda13_path(saved_cuda_home)
    env = os.environ.copy()
    if not venv.exists():
        run(["nvidia-smi"], env=env)
        run([sys.executable, mobile / "scripts/setup_m67_runtime.py",
             "--venv", venv, "--cuda", cuda, "--receipt", receipt], env=env)
    elif not python.is_file():
        raise RuntimeError(
            f"Partial isolated runtime exists at {venv} but its Python is missing. "
            "Preserve it; use a fresh Colab runtime rather than deleting or overwriting it."
        )

    if not python.is_file():
        raise FileNotFoundError(f"M67 runtime bootstrap did not create {python}")

    from sys import path as python_path

    python_path.insert(0, str(mobile / "scripts"))
    from setup_m64_runtime import runtime_env

    env = runtime_env(cuda, python.parent)
    runtime_probe = (
        "import json,sys; sys.path.insert(0,sys.argv[1]); "
        "from m64_teacher_qualification import environment; "
        "print(json.dumps(environment(),sort_keys=True))"
    )
    current_environment = json.loads(subprocess.check_output(
        [str(python), "-c", runtime_probe, str(mobile / "scripts")], env=env, text=True
    ))
    if current_environment != m67["environment"]:
        changes = {key: {"saved": m67["environment"].get(key), "current": current_environment.get(key)}
                   for key in sorted(set(m67["environment"]) | set(current_environment))
                   if m67["environment"].get(key) != current_environment.get(key)}
        raise RuntimeError(
            "Recreated runtime does not exactly match the signed M67 environment. "
            "No M67 manifest was changed; preserve files and inspect:\n" +
            json.dumps(changes, indent=2)
        )

    repo_state(repo)
    pre_patch_probe = (
        "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "import m64_teacher_qualification as q;import audit_m67_a2_coreml as m;"
        "print(q.source_hash(Path(sys.argv[2]))+' '+m.NATIVE_SOURCE_SHA)"
    )
    pre_patch_hash, native_hash = subprocess.check_output(
        [str(python), "-c", pre_patch_probe, str(mobile / "scripts"), str(repo)],
        env=env, text=True
    ).split()
    if pre_patch_hash not in {native_hash, m67["export_source_sha256"]}:
        raise RuntimeError(
            f"Existing MonoDETR source is not either reviewed M67 state: {pre_patch_hash}; "
            "preserve it without applying patches"
        )
    for script, extra in (
        ("patch_monodetr_colab_compat.py", ("--monodetr-repo", repo)),
        ("patch_monodetr_product_taxonomy.py", ("--monodetr-repo", repo)),
        ("patch_monodetr_mobilenetv4.py", ("--monodetr-repo", repo)),
        ("patch_m64_inference_sources.py", ("--repo", repo, "--role", "a2")),
    ):
        run([python, mobile / "scripts" / script, *extra], env=env)

    source_probe = (
        "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
        "import audit_m67_a2_coreml as m; import m64_teacher_qualification as q; "
        "print(q.source_hash(Path(sys.argv[2])))"
    )
    source_hash = subprocess.check_output(
        [str(python), "-c", source_probe, str(mobile / "scripts"), str(repo)], env=env, text=True
    ).strip()
    # The source may already contain the M67 Core ML patch. M67 prepare below
    # verifies/reapplies that exact patch and compares against the old manifest.
    probe_native_hash = subprocess.check_output(
        [str(python), "-c",
         "import sys;sys.path.insert(0,sys.argv[1]);import audit_m67_a2_coreml as m;print(m.NATIVE_SOURCE_SHA)",
         str(mobile / "scripts")], env=env, text=True
    ).strip()
    if source_hash not in {probe_native_hash, m67["export_source_sha256"]}:
        raise RuntimeError(f"Reconstructed M67 source differs from both reviewed states: {source_hash}")

    run([python, mobile / "scripts/build_m64_attention.py", "--repo", repo, "--role", "a2"], env=env)
    prepare = [python, mobile / "scripts/audit_m67_a2_coreml.py", "prepare",
               "--a2-manifest", args.m66_manifest.resolve(), "--repo", repo,
               "--output", output, "--dataset-root", Path(m67["dataset_root"]),
               "--split-dir", args.splits.resolve(), "--runtime-receipt", receipt]
    for candidate in args.candidate:
        prepare += ["--candidate", candidate.resolve()]
    run(prepare, cwd=mobile, env=env)
    verify_probe = (
        "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "import audit_m67_a2_coreml as m;m.load_manifest(Path(sys.argv[2]));"
        "print('M67 signed manifest, runtime, source, attention binary and fixed inputs verified')"
    )
    run([python, "-c", verify_probe, mobile / "scripts", manifest_path], cwd=mobile, env=env)
    print("M68 preflight ready: original A2/M67 identities reproduced; no training or M67 export rerun.", flush=True)


if __name__ == "__main__":
    main()
