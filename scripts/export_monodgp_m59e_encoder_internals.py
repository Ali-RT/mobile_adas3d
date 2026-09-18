"""Expose first-encoder internals from the frozen M59d TorchScript artifact.

No upstream checkout, dataset download, CUDA extension, or training is needed.
The in-memory graph gets diagnostic outputs; its arithmetic/weights stay fixed.
The graph interface is checked explicitly because TorchScript IR is internal API.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_monodgp_m59d_2d_transformer import (
    INPUT_SHAPES, PROBE_LIMIT, compare_tensors, operation_counts, sha256_file,
    tree_sha256,
)

TAP_NAMES = (
    "enc0_src", "enc0_pos", "enc0_reference_points", "enc0_query",
    "attn_value_projected", "attn_sampling_offsets", "attn_weight_logits",
    "attn_weight_probabilities", "attn_sampling_locations", "attn_weighted_sum",
    "enc0_attention_output", "enc0_residual1", "enc0_norm1",
    "enc0_ffn_linear1", "enc0_ffn_relu", "enc0_ffn_linear2",
    "enc0_residual2", "enc0_output",
)
PREFIX = "m59e__"


def _only(values, label):
    values = list(values)
    if len(values) != 1:
        raise RuntimeError(f"Unexpected M59d graph: {label} has {len(values)} candidates")
    return values[0]


def _call(graph, attribute):
    return _only((n for n in graph.nodes()
                  if n.kind() == "prim::CallMethod" and n.s("name") == "forward"
                  and n.inputsAt(0).node().kind() == "prim::GetAttr"
                  and n.inputsAt(0).node().s("name") == attribute), attribute)


def _tag(value, name):
    value.setDebugName(PREFIX + name)


def _tag_input(graph, torch, input_name, tap):
    value = _only((v for v in graph.inputs()
                   if v.debugName().split(".")[0] == input_name), input_name)
    # Route actual uses through a clone so the input remains observable on inline.
    first = next(graph.nodes())
    none = graph.create("prim::Constant")
    none.output().setType(torch._C.NoneType.get())
    none.insertBefore(first)
    clone = graph.create("aten::clone", [value, none.output()])
    clone.output().setType(value.type())
    clone.insertBefore(first)
    value.replaceAllUsesWith(clone.output())
    clone.replaceInput(0, value)
    _tag(clone.output(), tap)


def instrument_model(model, torch):
    """Instrument a freshly loaded (not yet executed) ScriptModule in memory."""
    layer = model.wrapped.det2d_transformer.encoder.layers._modules["0"]
    graph = layer.forward.graph
    for arg, tap in (("tensor", "enc0_src"), ("pos", "enc0_pos"),
                     ("reference_points", "enc0_reference_points")):
        _tag_input(graph, torch, arg, tap)
    attention_call = _call(graph, "self_attn")
    query = _only((n for n in graph.nodes() if n.kind() == 'aten::add'
                   and {n.inputsAt(0).debugName(), n.inputsAt(1).debugName()}
                   == {PREFIX + 'enc0_src', PREFIX + 'enc0_pos'}), 'source + position query')
    _tag(query.output(), "enc0_query")
    _tag(attention_call.output(), "enc0_attention_output")
    for attr, tap in (("norm1", "enc0_norm1"), ("linear1", "enc0_ffn_linear1"),
                      ("linear2", "enc0_ffn_linear2"), ("norm2", "enc0_output")):
        _tag(_call(graph, attr).output(), tap)
    _tag(_call(graph, "norm1").inputsAt(1), "enc0_residual1")
    _tag(_call(graph, "norm2").inputsAt(1), "enc0_residual2")
    relu = _only((n for n in graph.nodes() if n.kind() == "aten::relu"), "FFN ReLU")
    _tag(relu.output(), "enc0_ffn_relu")

    ag = layer.self_attn.forward.graph
    for attr, tap in (("value_proj", "attn_value_projected"),
                      ("sampling_offsets", "attn_sampling_offsets"),
                      ("attention_weights", "attn_weight_logits")):
        _tag(_call(ag, attr).output(), tap)
    softmax = _only((n for n in ag.nodes() if n.kind() == "aten::softmax"), "attention softmax")
    _tag(softmax.output(), "attn_weight_probabilities")
    locations = _only((v for n in ag.nodes() for v in n.outputs()
                       if v.debugName().split(".")[0] == "sampling_locations"), "sampling locations")
    _tag(locations, "attn_sampling_locations")
    _tag(_call(ag, "output_proj").inputsAt(1), "attn_weighted_sum")

    # Keep the ScriptModule method schema intact. Returning additional tensors
    # from its existing method can invalidate JIT's compiled return type.
    full = model.forward.graph.copy()
    torch._C._jit_pass_inline(full)
    original = _only(full.outputs(), "original output").node()
    if original.kind() != "prim::TupleConstruct":
        raise RuntimeError("Expected the M59d tuple output interface")
    original_outputs = list(original.inputs())
    values = [v for n in full.nodes() for v in n.outputs()]
    taps = [_only((v for v in values if v.debugName().split(".")[0] == PREFIX + name), name)
            for name in TAP_NAMES]
    outputs = original_outputs + taps
    result = full.create("prim::TupleConstruct", outputs)
    result.output().setType(torch._C.TupleType([v.type() for v in outputs]))
    full.appendNode(result)
    full.eraseOutput(0)
    full.registerOutput(result.output())
    torch._C._jit_pass_dce(full)
    torch._C._jit_pass_lint(full)
    return len(original_outputs), full


def internal_probe(graph, original_count, torch):
    """Prune the downstream model before tracing/converting the diagnostic."""
    graph = graph.copy()
    outputs = list(_only(graph.outputs(), "probe output").node().inputs())[original_count:]
    result = graph.create("prim::TupleConstruct", outputs)
    result.output().setType(torch._C.TupleType([v.type() for v in outputs]))
    graph.appendNode(result)
    graph.eraseOutput(0)
    graph.registerOutput(result.output())
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    return torch._C._create_function_from_graph("m59e_internal_probe", graph)


def verify_source(root):
    gate_path = root / "m59d_2d_transformer_export_gate.json"
    gate = json.loads(gate_path.read_text())
    if gate.get("complete") is not True or gate.get("all_export_gates_passed") is not True:
        raise RuntimeError("A passed M59d export is required")
    if gate.get("source_commit") != "aa059a18214aebf644510e7f0793971b403f9d14":
        raise RuntimeError("M59d upstream commit differs from the reviewed source")
    for name, field in (("MonoDGP_M59d_2d_transformer_fp32.pt", "torchscript_sha256"),
                        ("m59d_2d_transformer_reference_io.npz", "reference_io_sha256")):
        expected = gate.get("artifacts", {}).get(field)
        if not expected or sha256_file(root / name) != expected:
            raise RuntimeError(f"M59d missing or changed artifact: {name}")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m59d-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.m59d_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "m59e_encoder_internal_export_gate.json"
    package = output / "MonoDGP_M59e_encoder_internal_fp32.mlpackage"
    if package.exists() or report_path.exists():
        raise FileExistsError(f"Use a new output directory to preserve prior evidence: {output}")
    report = {"schema_version": 1, "complete": False,
              "experiment": "M59e first 2D encoder internal diagnostic export",
              "training_performed": False, "weights_changed": False,
              "physical_device_testing_authorized": False,
              "fp16_or_quantization_authorized": False, "product_safety_qualified": False}
    try:
        gate = verify_source(root)
        if np.lib.NumpyVersion(np.__version__) >= '2.4.0':
            raise RuntimeError('Core ML Tools 9 scalar conversion requires numpy<2.4; use the notebook dependency cell or a task-local environment')
        import torch
        import coremltools as ct
        torch.set_num_threads(4)
        path = root / "MonoDGP_M59d_2d_transformer_fp32.pt"
        names = gate["output_names"]
        with np.load(root / "m59d_2d_transformer_reference_io.npz", allow_pickle=False) as archive:
            inputs = tuple(torch.from_numpy(archive[n].copy()).float() for n in INPUT_SHAPES)
            baseline = {n: archive[n].copy() for n in names}
        if {n: list(t.shape) for n, t in zip(INPUT_SHAPES, inputs)} != INPUT_SHAPES:
            raise RuntimeError("M59d frozen input signature differs")
        print("M59e: source hashes verified; instrumenting encoder layer 0", flush=True)
        model = torch.jit.load(str(path), map_location="cpu").eval()
        original_count, graph = instrument_model(model, torch)
        probe = torch._C._create_function_from_graph("m59e_preservation_probe", graph)
        if original_count != len(names):
            raise RuntimeError("M59d output count differs from the graph")
        with torch.inference_mode():
            captured = probe(model, *inputs)
        preserved = {n: v.detach().numpy() for n, v in zip(names, captured[:original_count])}
        preservation = compare_tensors(baseline, preserved, names)
        report["original_m59d_output_preservation"] = preservation
        if not all(row["passed"] for row in preservation.values()):
            raise RuntimeError("M59e instrumentation did not preserve the M59d outputs")

        diagnostic = internal_probe(graph, original_count, torch)

        class Internals(torch.nn.Module):
            def __init__(self, source):
                super().__init__()
                self.source = source

            def forward(self, image, calibration, image_size):
                return diagnostic(self.source, image, calibration, image_size)

        wrapper = Internals(model).eval()
        reference = {n: v.detach().numpy() for n, v in zip(TAP_NAMES, captured[original_count:])}
        if len(reference) != len(TAP_NAMES) or not all(np.isfinite(v).all() for v in reference.values()):
            raise RuntimeError("M59e internal outputs are incomplete or non-finite")
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, inputs, strict=False, check_trace=False)
            replay = traced(*inputs)
        trace_parity = compare_tensors(reference, {n: v.numpy() for n, v in zip(TAP_NAMES, replay)}, TAP_NAMES)
        if not all(row["passed"] for row in trace_parity.values()):
            raise RuntimeError("M59e trace parity failed")
        print("M59e: original outputs and trace parity passed; converting FP32", flush=True)
        reference_path = output / "m59e_encoder_internal_reference_io.npz"
        np.savez_compressed(reference_path, **{n: t.numpy() for n, t in zip(INPUT_SHAPES, inputs)}, **reference)
        started = time.perf_counter()
        converted = ct.convert(traced, source="pytorch", convert_to="mlprogram",
                               minimum_deployment_target=ct.target.iOS17,
                               compute_precision=ct.precision.FLOAT32,
                               compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True,
                               inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float32)
                                       for n, t in zip(INPUT_SHAPES, inputs)],
                               outputs=[ct.TensorType(name=n) for n in TAP_NAMES])
        counts = operation_counts(converted._mil_program)
        if counts.get("custom", 0):
            raise RuntimeError("M59e conversion contains a custom operator")
        converted.save(str(package))
        report.update({"complete": True, "all_export_gates_passed": True,
                       "scope": "one frozen M59d input; instrumentation only",
                       "source_gate_sha256": sha256_file(root / "m59d_2d_transformer_export_gate.json"),
                       "source_torchscript_sha256": gate["artifacts"]["torchscript_sha256"],
                       "source_reference_io_sha256": gate["artifacts"]["reference_io_sha256"],
                       "torch_version": torch.__version__, "numpy_version": np.__version__,
                       "coremltools_version": ct.__version__,
                       "trace_parity": trace_parity, "probe_limit_max_abs": PROBE_LIMIT,
                       "output_names": list(TAP_NAMES),
                       "output_shapes": {n: list(v.shape) for n, v in reference.items()},
                       "conversion_seconds": time.perf_counter() - started,
                       "mil_operation_counts": counts,
                       "artifacts": {"mlpackage": str(package),
                                     "mlpackage_tree_sha256": tree_sha256(package),
                                     "reference_io": str(reference_path),
                                     "reference_io_sha256": sha256_file(reference_path)}})
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"M59e export report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
