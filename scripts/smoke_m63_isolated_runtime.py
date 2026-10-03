"""Exercise real dataset CUDA imports and kernels without optimizer updates."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from setup_m63_isolated_runtime import inventory, validate_inventory

REVISION = "M63H-SMOKE-2026-10-02-r5"


def loaded_native_libraries():
    """Record actual Linux mappings, not guesses from LD_LIBRARY_PATH."""
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return []
    names = ("libcuda", "libnvidia", "libnvvm", "libnvrtc", "libtorch",
             "MultiScaleDeformableAttention", "libllvmlite")
    paths = {line.split(maxsplit=5)[-1] for line in lines
             if any(name in line for name in names) and len(line.split(maxsplit=5)) == 6}
    return sorted(paths)


class SmokeTrace:
    """Flush a marker and durable progress before entering native code."""

    def __init__(self, path):
        self.path = Path(path)
        self.started = time.monotonic()
        self.events = []
        self.complete = False
        self.context = dict(
            revision=REVISION, pid=os.getpid(), python=sys.executable,
            smoke_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            python_version=sys.version,
            environment={key: os.environ.get(key) for key in (
                "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD",
                "NUMBA_CUDA_USE_NVIDIA_BINDING", "NUMBA_ENABLE_CUDASIM",
                "PYTHONPATH", "PYTHONNOUSERSITE")})

    def mark(self, stage, status, **details):
        event = dict(stage=stage, status=status,
                     elapsed_seconds=round(time.monotonic() - self.started, 3),
                     time_utc=datetime.now(timezone.utc).isoformat(), **details)
        self.events.append(event)
        print(f"[{REVISION}] {stage}: {status}", flush=True)
        if details:
            print(json.dumps(details, sort_keys=True), flush=True)
        progress = dict(self.context, complete=self.complete, optimizer_steps=0,
                        last_stage=stage, last_status=status, events=self.events,
                        loaded_native_libraries=loaded_native_libraries())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(progress, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    @contextmanager
    def stage(self, name):
        self.mark(name, "running")
        try:
            yield
        except Exception as exc:
            self.mark(name, "failed", error_type=type(exc).__name__, error=str(exc))
            raise
        else:
            self.mark(name, "passed")

    def finish(self):
        self.complete = True
        self.mark("complete", "passed")


def main():
    # Capture Python frames even when a C/CUDA library terminates the interpreter.
    # Enable before any third-party import; notebook also uses -X faulthandler.
    faulthandler.enable(all_threads=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    progress_path = a.output.with_suffix(".progress.json")
    trace = SmokeTrace(progress_path)
    trace.mark("startup", "passed", progress_file=str(progress_path))

    with trace.stage("runtime_inventory"):
        receipt = json.loads(a.receipt.read_text())
        current = inventory(sys.executable)
        validate_inventory(current)
        if current != receipt["inventory"]:
            raise RuntimeError("Runtime inventory changed")

    with trace.stage("frozen_manifest_load"):
        from m63_common import load_manifest
        from m61_common import sha256, write_json
        m = load_manifest(a.manifest)
        repo = Path(m["student"]["repo"])
        sys.path[:0] = [str(repo), str(repo / "lib/models/monodetr/ops")]

    # Preserve the original import/operation order; this is instrumentation only.
    with trace.stage("torch_numpy_import"):
        import torch
        import numpy as np
    with trace.stage("attention_extension_import"):
        import MultiScaleDeformableAttention
        print("CUDA extension:", MultiScaleDeformableAttention.__file__, flush=True)
    with trace.stage("kitti_dataset_import"):
        from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    with trace.stage("numba_iou_import"):
        from lib.datasets.kitti.kitti_eval_python.rotate_iou import rotate_iou_gpu_eval
    with trace.stage("numba_iou_kernel"):
        box = np.array([[0., 0., 2., 4., 0.]], dtype=np.float32)
        iou = rotate_iou_gpu_eval(box, box)
        if not np.allclose(iou, 1., atol=1e-5):
            raise RuntimeError("CUDA IoU smoke failed")
    with trace.stage("torch_cuda_backward"):
        x = torch.ones(4, device="cuda", requires_grad=True)
        (x * x).sum().backward()
        torch.cuda.synchronize()
        if not torch.equal(x.grad, torch.full_like(x, 2)):
            raise RuntimeError("CUDA backward failed")
    with trace.stage("write_smoke_report"):
        from numba import cuda
        report = dict(complete=True, optimizer_steps=0, dataset_import=True, iou=iou.tolist(),
            torch_cuda_backward=True, numba_cuda_module=cuda.__file__, revision=REVISION,
            runtime_receipt_sha256=sha256(a.receipt), manifest_sha256=m["manifest_sha256"],
            smoke_sha256=sha256(Path(__file__)),
            progress_file=str(progress_path), historical_runtime_equivalence_claimed=False)
        write_json(a.output, report)
    trace.finish()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
