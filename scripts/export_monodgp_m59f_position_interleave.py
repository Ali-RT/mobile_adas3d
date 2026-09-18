"""M59f: replace positional stack/flatten with explicit channel interleaving.

Reuses frozen TorchScript, without training, changing weights, or requiring
an upstream checkout. The encoder and full-model exports remain separate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_monodgp_m59d_2d_transformer import (
    INPUT_SHAPES, compare_tensors, operation_counts, sha256_file, tree_sha256,
)
from scripts.export_monodgp_m59e_encoder_internals import (
    TAP_NAMES, instrument_model, internal_probe, verify_source,
)
from scripts.validate_monodgp_m58_macos_parity import OUTPUT_NAMES


def rewrite_position_graph(graph, torch):
    """Rewrite exactly the two sine/cosine stack-flatten pairs in one method."""
    pairs = []
    for node in graph.nodes():
        if node.kind() != "aten::flatten":
            continue
        stack = node.inputsAt(0).node()
        if stack.kind() != "aten::stack":
            continue
        values = stack.inputsAt(0).node()
        if values.kind() != "prim::ListConstruct":
            continue
        operands = list(values.inputs())
        if [value.node().kind() for value in operands] != ["aten::sin", "aten::cos"]:
            continue
        if (stack.inputsAt(1).toIValue() != 4 or node.inputsAt(1).toIValue() != 3
                or node.inputsAt(2).toIValue() != -1):
            raise RuntimeError("Unexpected positional stack/flatten dimensions")
        pairs.append((node, values.output()))
    # Frozen PositionEmbeddingSine has 128 coordinates in each y/x block.
    aranges = [node for node in graph.nodes() if node.kind() == "aten::arange"]
    if len(pairs) != 2 or len(aranges) != 1 or aranges[0].inputsAt(0).toIValue() != 128:
        raise RuntimeError("Expected two positional pairs and 128-coordinate encoding")
    indices = torch.tensor([value for i in range(64) for value in (i, i + 64)], dtype=torch.long)
    for flatten, operands in pairs:
        with graph.insert_point_guard(flatten):
            dim = graph.insertConstant(3)
            index = graph.insertConstant(indices)
        concatenated = graph.create("aten::cat", [operands, dim])
        concatenated.output().setType(flatten.output().type())
        concatenated.insertBefore(flatten)
        selected = graph.create("aten::index_select", [concatenated.output(), dim, index])
        selected.output().setType(flatten.output().type())
        selected.insertBefore(flatten)
        flatten.output().replaceAllUsesWith(selected.output())
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    return len(pairs)


def rewrite_positions(model, torch):
    """Only touch the four traced PositionEmbeddingSine methods; reject drift."""
    modules = [(name, module) for name, module in model.named_modules()
               if module.original_name == "PositionEmbeddingSine"]
    if len(modules) != 1:
        raise RuntimeError("Expected one PositionEmbeddingSine module")
    name, module = modules[0]
    methods = module._c._method_names()
    if set(methods) != {"forward", "forward1", "forward2", "forward3"}:
        raise RuntimeError(f"Unexpected positional methods: {methods}")
    return {f"{name}.{method}": rewrite_position_graph(module._c._get_method(method).graph, torch)
            for method in methods}


def verify_full_source(root):
    path = root / "m58_coreml_export_gate.json"
    gate = json.loads(path.read_text())
    if gate.get("complete") is not True or gate.get("all_export_gates_passed") is not True:
        raise RuntimeError("Passed M58 export gate required")
    if gate.get("output_names") != list(OUTPUT_NAMES):
        raise RuntimeError("M58 full-model interface changed")
    if gate.get("source_checkpoint_sha256") != "7d18883d6f998e7616beaa45b92d348b22728f3fad87a891618b5ce5a1cde17c":
        raise RuntimeError("M58 checkpoint differs from the reviewed M56d parent")
    for name, key in (("MonoDGP_M58_fixed_fp32.pt", "torchscript_sha256"),
                      ("m58_reference_io.npz", "reference_io_sha256")):
        if sha256_file(root / name) != gate["artifacts"][key]:
            raise RuntimeError(f"M58 source artifact changed: {name}")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("internals", "layers", "full"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Frozen M59d directory for internals/layers; M58 for full")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.source_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "m59f_export_gate.json"
    package = output / "MonoDGP_M59f_fp32.mlpackage"
    if report_path.exists() or package.exists():
        raise FileExistsError("Use a new output directory to preserve evidence")
    report = {"schema_version": 1, "complete": False, "stage": args.stage,
              "experiment": "M59f explicit positional channel interleaving",
              "training_performed": False, "weights_changed": False,
              "compute_precision": "FLOAT32", "deployment_authorized": False}
    try:
        if np.lib.NumpyVersion(np.__version__) >= "2.4.0":
            raise RuntimeError("Use numpy<2.4 with Core ML Tools 9")
        import torch
        import coremltools as ct
        torch.set_num_threads(4)
        if args.stage == "full":
            gate = verify_full_source(root)
            stem, gate_name, io_name = "MonoDGP_M58_fixed_fp32", "m58_coreml_export_gate.json", "m58_reference_io.npz"
        else:
            gate = verify_source(root)
            stem, gate_name, io_name = "MonoDGP_M59d_2d_transformer_fp32", "m59d_2d_transformer_export_gate.json", "m59d_2d_transformer_reference_io.npz"
        trace_path = root / (stem + ".pt")
        with np.load(root / io_name, allow_pickle=False) as archive:
            inputs = tuple(torch.from_numpy(archive[name].copy()).float() for name in INPUT_SHAPES)
            reference = {name: archive[name].copy() for name in gate["output_names"]}
        if {name: list(value.shape) for name, value in zip(INPUT_SHAPES, inputs)} != INPUT_SHAPES:
            raise RuntimeError("Frozen input shapes changed")
        names = gate["output_names"]
        if args.stage == "internals":
            original = torch.jit.load(str(trace_path), map_location="cpu").eval()
            count, graph = instrument_model(original, torch)
            probe = internal_probe(graph, count, torch)
            with torch.inference_mode():
                values = probe(original, *inputs)
            names = list(TAP_NAMES)
            reference = {name: value.numpy() for name, value in zip(names, values)}
            del original, graph, probe, values
        model = torch.jit.load(str(trace_path), map_location="cpu").eval()
        report["rewritten_pairs_by_method"] = rewrite_positions(model, torch)
        if args.stage == "internals":
            count, graph = instrument_model(model, torch)
            probe = internal_probe(graph, count, torch)

            class Wrapper(torch.nn.Module):
                def __init__(self, source):
                    super().__init__()
                    self.source = source

                def forward(self, image, calibration, image_size):
                    return probe(self.source, image, calibration, image_size)

            with torch.inference_mode():
                model = torch.jit.trace(Wrapper(model).eval(), inputs, check_trace=False, strict=False)
        with torch.inference_mode():
            rewritten = model(*inputs)
        candidate = {name: value.numpy() for name, value in zip(names, rewritten)}
        preservation = compare_tensors(reference, candidate, names)
        report["pytorch_preservation"] = preservation
        if not all(row["passed"] for row in preservation.values()):
            raise RuntimeError("Export rewrite changed PyTorch outputs beyond 0.001")
        print(f"M59f {args.stage}: PyTorch preservation passed; converting FP32", flush=True)
        converted = ct.convert(model, source="pytorch", convert_to="mlprogram",
                               minimum_deployment_target=ct.target.iOS17,
                               compute_precision=ct.precision.FLOAT32,
                               compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True,
                               inputs=[ct.TensorType(name=name, shape=value.shape, dtype=np.float32)
                                       for name, value in zip(INPUT_SHAPES, inputs)],
                               outputs=[ct.TensorType(name=name) for name in names])
        counts = operation_counts(converted._mil_program)
        if counts.get("custom", 0):
            raise RuntimeError("Custom MIL operators remain")
        converted.save(str(package))
        io_path = output / "m59f_reference_io.npz"
        np.savez_compressed(io_path, **{name: value.numpy() for name, value in zip(INPUT_SHAPES, inputs)}, **reference)
        report.update({"complete": True, "all_export_gates_passed": True,
                       "source_gate_sha256": sha256_file(root / gate_name),
                       "source_torchscript_sha256": sha256_file(trace_path),
                       "source_reference_io_sha256": sha256_file(root / io_name),
                       "torch_version": torch.__version__, "numpy_version": np.__version__,
                       "coremltools_version": ct.__version__, "output_names": names,
                       "output_shapes": {name: list(value.shape) for name, value in reference.items()},
                       "mil_operation_counts": counts,
                       "artifacts": {"mlpackage_tree_sha256": tree_sha256(package),
                                     "reference_io_sha256": sha256_file(io_path)}})
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("M59f export report:", report_path, flush=True)


if __name__ == "__main__":
    main()
