# Parking range preliminary model screen

The new priority is Vehicle and Pedestrian labeling and distance estimates at
2 to 6 metres for ultrasonic validation from a moving, mounted camera. The
version 0.1 criteria and workflow are recorded in the
[parking range contract](../../PARKING_RANGE_VALIDATION_CONTRACT.md).

The first local analysis reused all 3,769 saved trained A2-Lite optimized-graph
prediction files and the frozen Chen validation labels. It performed no model
inference, training or phone changes. The complete full A2 prediction directory
is not available locally, so the paired comparison remains pending.

## Available reference examples

| Camera center depth band | Vehicle objects | Pedestrian objects |
| --- | ---: | ---: |
| 2 to 3 m | 159 | 11 |
| 3 to 4 m | 271 | 37 |
| 4 to 5 m | 210 | 65 |
| 5 to 6 m | 243 | 102 |

All 883 Vehicle targets in these bands have nonzero truncation. Of the 215
Pedestrian targets, 210 have nonzero truncation. These examples are therefore
not a representative mounted parking benchmark, and the three closer
Pedestrian bands fall below the provisional 100-object evidence floor.

## Trained A2 Lite results

The table uses correct-class 2D matching at IoU at least 0.5 and the historical
serialized score cutoff 0.001. Recall includes all reference objects in each
band. Distance errors are descriptive statistics of valid matched distances;
misses remain in the recall denominator and in the delivered object table.

| Class | Camera center depth band | Matched targets | Recall | Median absolute depth error | p95 absolute depth error |
| --- | --- | ---: | ---: | ---: | ---: |
| Vehicle | 2 to 3 m | 5 / 159 | 3.1% | 3.85 m | 8.35 m |
| Vehicle | 3 to 4 m | 13 / 271 | 4.8% | 3.98 m | 7.69 m |
| Vehicle | 4 to 5 m | 59 / 210 | 28.1% | 1.49 m | 4.22 m |
| Vehicle | 5 to 6 m | 145 / 243 | 59.7% | 0.41 m | 2.03 m |
| Pedestrian | 2 to 3 m | 4 / 11 | 36.4% | 1.53 m | 1.64 m |
| Pedestrian | 3 to 4 m | 15 / 37 | 40.5% | 0.85 m | 2.99 m |
| Pedestrian | 4 to 5 m | 41 / 65 | 63.1% | 0.37 m | 1.41 m |
| Pedestrian | 5 to 6 m | 86 / 102 | 84.3% | 0.40 m | 1.32 m |

All four Vehicle bands fail the populated recall check. The 5 to 6 metre
Pedestrian band also fails. The three closer Pedestrian bands are
inconclusive under the evidence floor, not passing. Distance statistics based
on fewer than 50 matches are descriptive rather than supported distance-gate
decisions. The machine-readable summary also includes precision, bias,
coverage, truncation, evidence counts and individual threshold checks.

## Interpretation and next action

This is a camera-forward-depth screen of KITTI annotations, not a measurement
of reflecting-surface range from the ultrasonic sensor. A long nearby vehicle
can have a box center far from its closest visible surface. Neither the
screen's failures nor a possible future pass directly certify the mounted
parking use case. The current Lite model nevertheless has not demonstrated
the declared near-range accuracy.

Run the new CPU-only Colab notebook against the complete saved M68 A2 and
trained Lite prediction directories. Then collect independently referenced,
synchronized mounted-camera measurements in the actual 2 to 6 metre surface
range. Keep KD, further shrinking and changes to historical targets separate
from this measurement task.

Local evidence: `outputs/parking_range_validation_20261021/lite_r1/` contains
the input fingerprint, frozen contract, per-object matches and misses,
prediction assignments, class and band tables, JSON summary and results ZIP.
The Lite checkpoint is `0cd8dd0d…e176930` and optimized ONNX graph is
`47e9b9af…cc1db8`; full hashes are retained in the input manifest. The manifest
records the source model identity, but model provenance alone is not a
cryptographic binding of predictions to an inference run.
