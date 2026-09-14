from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"
TARGET = Path("lib/models/monodgp/ops/modules/ms_deform_attn.py")
MARKER = "def ms_deform_attn_core_portable("
MODE_ENVIRONMENT_VARIABLE = "MONODGP_PORTABLE_DEFORM_ATTN"


PORTABLE_IMPLEMENTATION = r'''

# M57 rank-five deformable-attention decomposition. Native CUDA remains the
# default path; the portable branch is enabled explicitly for parity/export.
M57_PORTABLE_SPATIAL_SHAPES = ((48, 160), (24, 80), (12, 40), (6, 20))


def ms_deform_attn_core_portable(
        value, sampling_locations, attention_weights, spatial_shapes):
    """Decompose multi-scale deformable attention into native PyTorch ops."""
    batch, _, heads, channels = value.shape
    query_count = sampling_locations.shape[1]
    points = sampling_locations.shape[3] // len(spatial_shapes)
    split_sizes = tuple(height * width for height, width in spatial_shapes)
    values = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for level, (height, width) in enumerate(spatial_shapes):
        value_level = (
            values[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, channels, height, width)
        )
        start = level * points
        grid_level = (
            sampling_grids[:, :, :, start:start + points]
            .transpose(1, 2)
            .flatten(0, 1)
        )
        sampled.append(
            F.grid_sample(
                value_level,
                grid_level,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads, 1, query_count, len(spatial_shapes) * points
    )
    output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    output = output.reshape(batch, heads * channels, query_count)
    return output.transpose(1, 2).contiguous()


def ms_deform_attn_portable_forward(
        module, query, reference_points, value, input_spatial_shapes,
        input_level_start_index):
    """Build rank-five sampling tensors and execute the portable core."""
    spatial_shapes = module.portable_spatial_shapes
    if not torch.jit.is_tracing():
        actual_shapes = tuple(
            tuple(int(item) for item in row.tolist())
            for row in input_spatial_shapes.detach().cpu()
        )
        expected_starts = []
        running = 0
        for height, width in spatial_shapes:
            expected_starts.append(running)
            running += height * width
        actual_starts = tuple(
            int(item) for item in input_level_start_index.detach().cpu().tolist()
        )
        if actual_shapes != spatial_shapes:
            raise RuntimeError(
                "M57 portable attention requires spatial shapes {} but got {}"
                .format(spatial_shapes, actual_shapes)
            )
        if actual_starts != tuple(expected_starts):
            raise RuntimeError(
                "M57 portable attention level starts changed: {} != {}"
                .format(actual_starts, tuple(expected_starts))
            )

    batch, query_count, _ = query.shape
    flattened_tokens = value.shape[1]
    expected_tokens = sum(height * width for height, width in spatial_shapes)
    if not torch.jit.is_tracing() and flattened_tokens != expected_tokens:
        raise RuntimeError(
            "M57 portable attention expected {} tokens, got {}"
            .format(expected_tokens, flattened_tokens)
        )
    level_points = module.n_levels * module.n_points
    offsets = module.sampling_offsets(query).reshape(
        batch, query_count, module.n_heads, level_points, 2
    )
    weights = F.softmax(
        module.attention_weights(query).reshape(
            batch, query_count, module.n_heads, level_points
        ),
        dim=-1,
    )
    centers = reference_points[..., :2].repeat_interleave(
        module.n_points, dim=2
    ).unsqueeze(2)
    if reference_points.shape[-1] == 2:
        normalizer = query.new_tensor(
            [[width, height] for height, width in spatial_shapes]
        ).repeat_interleave(module.n_points, dim=0)
        locations = centers + offsets / normalizer[None, None, None]
    elif reference_points.shape[-1] == 6:
        scale = (
            reference_points[..., 2::2] + reference_points[..., 3::2]
        ) * 0.5
        scale = scale.repeat_interleave(module.n_points, dim=2).unsqueeze(2)
        locations = centers + offsets / module.n_points * scale
    else:
        raise ValueError(
            "M57 portable attention requires 2D or 6D reference points"
        )
    return ms_deform_attn_core_portable(
        value, locations, weights, spatial_shapes
    )
'''


def patch_source(source: str) -> tuple[str, str]:
    if MARKER in source:
        required = (
            "self.use_portable_deform_attn",
            "ms_deform_attn_portable_forward(",
            MODE_ENVIRONMENT_VARIABLE,
            "F.grid_sample(",
        )
        if not all(item in source for item in required):
            raise RuntimeError("Incomplete pre-existing M57 patch")
        return source, "already_patched"

    prerequisites = (
        "from torch.nn import Linear as _LinearWithBias",
        "from torch import overrides as torch_overrides",
        "class MSDeformAttn(nn.Module):",
        "output = MSDeformAttnFunction.apply(",
    )
    if not all(item in source for item in prerequisites):
        raise RuntimeError(
            "Apply patch_monodgp_colab_compat.py before the M57 patch"
        )

    future_anchor = "from __future__ import division\n\nimport warnings"
    if source.count(future_anchor) != 1:
        raise RuntimeError("Unexpected MonoDGP import anchor")
    source = source.replace(
        future_anchor,
        "from __future__ import division\n\nimport os\nimport warnings",
        1,
    )

    class_anchor = "\n\nclass MSDeformAttn(nn.Module):"
    if source.count(class_anchor) != 1:
        raise RuntimeError("Unexpected MonoDGP MSDeformAttn class anchor")
    source = source.replace(
        class_anchor,
        PORTABLE_IMPLEMENTATION + class_anchor,
        1,
    )

    init_anchor = (
        "        self.n_points = n_points\n\n"
        "        self.sampling_offsets = nn.Linear("
    )
    if source.count(init_anchor) < 2:
        raise RuntimeError("Unexpected MonoDGP attention initialization anchor")
    source = source.replace(
        init_anchor,
        "        self.n_points = n_points\n"
        "        self.use_portable_deform_attn = (\n"
        "            os.environ.get(\"MONODGP_PORTABLE_DEFORM_ATTN\") == \"1\"\n"
        "        )\n"
        "        self.portable_spatial_shapes = M57_PORTABLE_SPATIAL_SHAPES\n\n"
        "        self.sampling_offsets = nn.Linear(",
        1,
    )

    forward_anchor = (
        "        if self.conditional:\n"
        "            value = value.view(N, Len_in, self.n_heads, (self.d_model//2) // self.n_heads)\n"
        "        else:\n"
        "            value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)\n"
        "        sampling_offsets = self.sampling_offsets(query).view("
    )
    if source.count(forward_anchor) != 1:
        raise RuntimeError("Unexpected MonoDGP MSDeformAttn forward anchor")
    source = source.replace(
        forward_anchor,
        "        if self.conditional:\n"
        "            value = value.view(N, Len_in, self.n_heads, (self.d_model//2) // self.n_heads)\n"
        "        else:\n"
        "            value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)\n"
        "        if self.use_portable_deform_attn:\n"
        "            output = ms_deform_attn_portable_forward(\n"
        "                self, query, reference_points, value,\n"
        "                input_spatial_shapes, input_level_start_index)\n"
        "            return self.output_proj(output)\n"
        "        sampling_offsets = self.sampling_offsets(query).view(",
        1,
    )

    if source.count(MARKER) != 1:
        raise RuntimeError("M57 portable core marker missing after patch")
    if source.count("self.use_portable_deform_attn") != 2:
        raise RuntimeError("M57 mode flag was not patched exactly once")
    if source.count("output = MSDeformAttnFunction.apply(") != 2:
        raise RuntimeError("M57 unexpectedly changed a native CUDA call site")
    return source, "patched"


def patch_monodgp(repo: Path) -> str:
    repo = repo.resolve()
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP {PINNED_COMMIT}, found {commit}")
    patched, result = patch_source(target.read_text(encoding="utf-8"))
    if result == "patched":
        target.write_text(patched, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add an opt-in rank-five deformable-attention path to MonoDGP."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    args = parser.parse_args()
    result = patch_monodgp(args.monodgp_repo)
    print(f"MonoDGP M57 deformable-attention patch: {result}")
    print(f"Pinned commit: {PINNED_COMMIT}")
    print(f"Mode environment variable: {MODE_ENVIRONMENT_VARIABLE}=1")


if __name__ == "__main__":
    main()
