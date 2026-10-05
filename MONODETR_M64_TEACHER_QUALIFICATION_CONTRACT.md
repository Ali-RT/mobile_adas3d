# M64 A2 baseline and teacher qualification

Prepared 2026-10-04. This experiment checks whether unchanged A2 has a useful,
reproducible stronger teacher before another distillation attempt. It performs
zero optimizer updates. A2 remains selected; MonoPRIO is a candidate only.
Local regression and pinned-source import checks do not establish CUDA execution,
published accuracy reproduction, improvement or deployment readiness.

## Models and scope

The student remains MobileNetV4 Conv Medium plus the full MonoDETR graph,
with its exact GT-trained epoch130 checkpoint:
`ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4`.
Its original product moderate 3D AP_R40 is Vehicle 15.4573 and Pedestrian
7.5328. Neither weights nor inference architecture are modified.

The single teacher candidate is the unified native Car/Pedestrian/Cyclist
validation model in the [official MonoPRIO repository](https://github.com/Leon-Davies/MonoPRIO).
Pin upstream commit `884b7d8562e528031616c4773170bfc4fe211bf0` and use its
seed444 validation checkpoint, validation prior bank and published log. Do not
substitute a trainval/test checkpoint. The published seed444 3D AP_R40 values
are Car 30.4130/21.9274/18.7024 and Pedestrian 12.4075/9.6523/7.4288
(easy/moderate/hard), not the median of five seeds or our product benchmark.

Native Car is not our Vehicle class, which also includes Van, Truck and Tram;
native Pedestrian does not establish coverage of Person_sitting. Report both
native-class and product-taxonomy metrics without relabeling their protocols.
No full product adaptation or new teacher training is included.

## Runtime and source boundaries

Use a new private virtualenv with PyTorch 2.10.0, torchvision 0.25.0 and CUDA
13.0, the [official compatible wheel pair](https://pytorch.org/get-started/previous-versions/).
The notebook neither changes the Colab kernel packages nor restores the old
M62 environment. It binds this prospective runtime and source identity to a
new RUN_ID. Keep the same recorded identity for cache resumption; preserve the
folder and choose a new RUN_ID after a GPU/runtime change.

The A2 upstream pin remains `6994b9f512400b258c6edb75f77423beb9c126f2`.
Apply the existing compatibility, product-taxonomy and MobileNetV4 patches only
to new M64 checkouts. Both loaders import AP evaluators only inside their eval
methods. A2 and teacher run in separate processes with separate local attention
binaries. The published native evaluator runs in another process without prior
Torch CUDA initialization. A native evaluator crash cannot authorize KD or
discard completed prediction caches.

Narrow teacher fixes repair its class-docstring indentation, prohibit pickle
loading of numeric prior arrays, and suppress unused ImageNet initialization
downloads before strict full-checkpoint loading. These do not replace model
operators or change trained inference behavior. Frozen M62/M63 sources, losses,
manifests and artifacts must not be rewritten.

## Six notebook sections

1. Settings, main-branch project update, revision checks and durable logs.
2. Private runtime, pinned sources, narrow patches, local CUDA builds and tests.
3. Original KITTI data/splits, official asset acquisition and prospective identity.
4. Restricted weight loading and finite forward checks for both models; one A2
   GT backward with unchanged parameters and frozen running buffers. No optimizer
   is created. This is not proof of a successful continuation recipe.
5. All 3,769 Chen validation images, atomic resumable per-image caches, independent
   product and native-class metrics, nearby/geometry diagnostics and published
   native teacher evaluation in a separate process.
6. Review report and `m64_results.zip`; stop before training or distillation.

Run sections 1–6 in order on a GPU. No iPhone connection is needed. After an
interruption, rerun in order; verified setup/assets/cache records are reused.
Publish the notebook and all four supporting scripts together to `main` before
running in Colab; section 1 checks the notebook/script revision and will stop
if the checkout does not contain the matching workflow.
Section 6 can bundle a failed run's available diagnostics. If data restoration
reports missing files, supply the original KITTI folders at a listed candidate
path; this workflow does not download the KITTI dataset.

## Evidence and fixed checks

Train/val IDs must be exactly Chen 3712/3769 in canonical order, with no overlap.
Normalize line endings for ID verification: the teacher's released split files
use CRLF but their IDs match the canonical split. Bind original labels and
calibration, model/config/source identities, built binaries, asset hashes and
actual transformed inference inputs. Do not silently recompute a changed cache.

The release does not publish independent asset checksums. The workflow pins
hashes of the actual release bytes downloaded and inspected locally on 2026-10-04;
require each download to match them. This is not publisher-signed provenance.
Restricted CPU checkpoint loading, strict model-state loading and all seven
numeric prior buffers passed local inspection. CUDA-only operator imports and
the loss constructor were stubbed for that CPU state check; no GPU inference,
loss execution or accuracy reproduction was established. Require prior buffers to match
the checkpoint's fixed router buffers. The bank format does not independently
identify all construction sample IDs. Matching buffers alone cannot prove that
validation labels were excluded; prior provenance remains a manual review item.

Unchanged A2 numerical/protocol reproduction allows at most 0.15 AP points on
each of the four original moderate 3D/BEV entries and 0.01 absolute nearby recall
drift per class. These are prospective reproduction checks, not relaxed product
targets. Published native teacher reproduction allows at most 0.5 AP points on
each of its six published 3D entries, fixed before observing the run.

Product/native-independent predictions use class threshold 0.001 and top50.
Published teacher reproduction uses its native threshold 0.2 and top50. The
upstream decoder multiplies confidence by depth uncertainty after class-score
filtering; retain that convention and native two-decimal KITTI export. The
independent evaluator is not a substitute for published native reproduction.
Nearby recall uses class-correct 2D IoU >=0.5 matches, Vehicle<40m and
Pedestrian<30m, at exported score>=0.001. It is not a 3D correctness rate.

## Review decision and later work

Return identities, smoke reports, complete-set AP tables, geometry/nearby CSVs,
official log and durable execution logs. Large weights and per-image predictions
remain on Drive. A failed native process must be reported, not converted into
an accuracy pass. Historical product gates remain unchanged.

No notebook outcome automatically selects the teacher or authorizes KD.
Review A2 stability, native reproduction, prior provenance and actual supported
class/range advantages. If a teacher is useful, the next bounded experiment is
GT plus original-A2 preservation control. After that is stable, compare the same
control with one object-matched feature-KD treatment, retaining both classes'
GT and preservation supervision. A promising result needs seed confirmation
before longer training. No automatic teacher search, temperature grid, scalar-
depth retry, compression or phone experiment follows a failed M64 check.
