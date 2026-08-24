# MobileADAS3D-H2 spatial-reference query contract

Status: implementation preflight; training is not yet authorized.

## Purpose

H2 addresses H1-v2's unstable Hungarian ownership without changing the
qualified mobile backbone or expanding the exported output interface. H1
remains frozen and reproducible.

## Frozen graph delta

- Keep MobileNetV4 Conv Small, Lite-FPN, depth context, transformer width 192,
  two encoder layers, two decoder layers, six attention heads, and 50 queries.
- Arrange the 50 queries on a fixed row-major 10-column × 5-row normalized
  image grid. Cell centers span x=`0.05..0.95` and y=`0.10..0.90`.
- Add the matching fixed 2D sine/cosine encoding to each learned content query.
- Interpret box-center and projected-center head values as offsets from the
  query reference point: `reference + 0.10 * tanh(raw_offset)`, clamped to
  `[0,1]`.
- Keep width/height decoding, all remaining heads, and all nine exported tensor
  names and shapes unchanged.
- Keep the H1-v2 implicit-background loss and Hungarian cost for the first H2
  capacity experiment. Distillation remains disabled.

## Required gates

1. Unit forward/backward, finite outputs, exact output names/shapes, and H1
   regression tests.
2. Single-image 1,000-step capacity gate using the unchanged thresholds.
3. Fresh Tiny16 gate with query-assignment stability included in the report.
4. Only after both capacity gates pass: Core ML parity/physical-device graph
   requalification, followed by a bounded GT-only full-data gate.

Failure of the H2 capacity gates blocks full KITTI training and distillation.
