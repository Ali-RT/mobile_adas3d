# Parking range model validation

The current use case is Vehicle and Pedestrian labeling and distance estimates
from a camera mounted on a slowly moving vehicle, to support ultrasonic sensor
validation. The critical range is 2 to 6 metres. This separate workflow compares
the existing full A2 teacher and the trained label-only A2-Lite epoch 10 model.
It does not train, distill, change model weights, or modify historical gates.

## Provisional criteria

These version 0.1 engineering targets were declared on 21 October 2026 before
the new range analysis. They are not an industry standard or a sensor
certification. The machine-readable contract is
[`configs/parking_range_validation_v0_1.json`](configs/parking_range_validation_v0_1.json).

| Measurement | Target |
| --- | ---: |
| Median absolute distance error | At most 0.10 m |
| p95 absolute distance error | At most 0.20 m |
| Absolute mean signed distance error | At most 0.05 m |
| Correct-class detection recall | At least 0.95 |
| Correct-class detection precision | At least 0.95 |

As a completeness guard, at least 95 percent of target objects must also
receive a correctly matched, finite, positive-geometry distance. This is
reported separately from detection recall so an invalid range cannot be
silently excluded from every distance statistic.

Apply the criteria to each class separately in 2 to 3, 3 to 4, 4 to 5 and
5 to 6 metre bands. A boundary belongs to the higher band; 6 metres belongs
to the last band. No overall average can substitute for a missing or failing
class and band. Do not choose a new threshold after seeing which model wins.

## First stage using saved KITTI predictions

Run the CPU-only notebook
[`MonoDETR_Parking_2_to_6m_Screen_Colab.ipynb`](notebooks/MonoDETR_Parking_2_to_6m_Screen_Colab.ipynb).
It reuses decoded predictions from the complete frozen Chen validation split.
Only labels and small prediction text files are staged, with four copy workers.
No images, GPU allocation, native extension, model download, or historical
Python environment are needed. Missing source predictions are reported, not
replaced by a different model or silently regenerated.

This stage measures **camera forward depth to the annotated 3D box center**.
It does not measure the reflecting surface relative to the ultrasonic sensor.
A numeric KITTI screen pass therefore cannot qualify the parking measurement
system. The output always leaves parking and ultrasonic qualification false.
A visible, long vehicle may have its annotated box center outside this range,
even behind the camera. Such targets remain in 2D matching and their
nonpositive camera-center depths are counted, not treated as corrupt labels.

Match predictions by descending serialized score, correct product class, and
2D IoU at least 0.5, with one prediction per target. Match against all target
depths before filtering to the 2 to 6 metre ground-truth interval. Otherwise a
bad depth estimate could disappear from the evaluation. Occluded and truncated
targets remain included, with their counts reported. DontCare regions do not
suppress false positives in this conservative preliminary screen.

Recall and distance statistics use the ground-truth depth band. Precision uses
the predicted depth band: a prediction counts as correct only when it matches
a correctly labeled target whose true depth is anywhere within 2 to 6 metres.
This avoids requiring predictions to remain in an identical one-metre bin
across a boundary. Unmatched predictions with invalid or missing depth cannot
be assigned a range and prevent a precision pass. These definitions are
explicitly diagnostic and are not a replacement for sensor-frame evaluation.

The score cutoff is the historical serialized, uncertainty-reweighted score
0.001. This is not the phone's pre-reweighting class-confidence cutoff; the
KITTI files do not retain that separate value. Report that limitation rather
than claiming the screen reproduces phone eligibility exactly.

For an initial screen, each class and band needs 100 target observations from
at least 25 frames; distance statistics need 50 valid matched distances and
precision needs 100 predicted observations in its band. These are provisional
evidence floors, not a confidence guarantee. Correlated frames are not
independent trials. Small samples produce an **inconclusive** screen, not a
pass. A sufficiently populated recall failure remains a failure even if too
few distances were returned. Missing distances remain visible as failures of
coverage rather than becoming zero-error measurements.

Hashes bind the split, contract, labels, predictions and implementation to
each new result. Optional model manifests verify the declared checkpoint and
ONNX identity, but do not by themselves prove that supplied prediction files
came from that graph. Prediction source directories remain read-only. Reuse
of an output directory is allowed only for byte-identical results.

## Mounted parking validation

Before a parking pass can be declared, collect independently referenced
measurements at 2, 3, 4, 5 and 6 metres and between those distances. Record
camera intrinsics, crop, camera-to-sensor mounting transform, measurement
timestamps, clock alignment, vehicle speed, object identity and reference
uncertainty. Compare the same physical target surface, not a camera-relative
box-center depth against an acoustic surface range.

Surveyed target placements or another qualified independent reference must
establish the distance. The ultrasonic sensor under test must not supply its
own ground truth. Quantify reference, calibration and synchronization
uncertainty before making a threshold decision. Insufficient reference quality
is inconclusive. The minimum evidence floors above do not establish coverage
of all parking conditions or qualify a safety function.

The acceptable sampling rate and processing delay remain open until maximum
vehicle speed and event coverage are specified. Delayed outputs may be useful
when paired with measurements from the original capture time; this does not
make sparse sampling, clock error or thermal shutdown acceptable by default.
The historical 50 ms and 10 FPS results retain their original meaning.

## Commands and deliverables

Use the `inventory` stage to count available labels without model predictions.
Use `screen` with one or both model prediction directories. The notebook has
explicit paths for the saved M68 A2 and trained A2-Lite results, followed by a
single source preflight. It never selects the newest folder automatically.

Each run writes a JSON summary, class and range CSV tables, per-object matches
and misses, per-prediction assignments, an input fingerprint and a results ZIP.
All are analysis records, not a trained model or phone deployment approval.
The weekly report dated 20 October remains a historical report and is not
rewritten by this workflow.

Measurement background: [NIST measurement uncertainty guidance](https://www.nist.gov/pml/nist-technical-note-1297)
and [TI ultrasonic sensing basics](https://www.ti.com/lit/pdf/slaa907).
