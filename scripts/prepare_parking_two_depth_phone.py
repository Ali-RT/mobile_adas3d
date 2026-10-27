"""Export the unchanged metric-outdoor depth challenger for camera-OFF iPhone A/B.

Shared YOLO26n detector, ORT CPU-2, fixed 720p-shaped engineering inputs. This
does not validate native camera preprocessing, surface accuracy, or live use.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import parking_edge_common as common
from scripts import compare_parking_depth_models as comparison
from scripts.prepare_parking_candidates_phone import checked_archive, INPUT_SHA, MANIFEST_SHA, SPECS, letterbox
from scripts.prepare_parking_depth_phone import GRAPH_SHA, preprocess

REVISION="PARKING-TWO-DEPTH-PHONE-2026-10-26-r1"
INPUT_SHAPE=[1,3,518,924]
OUTPUT_SHAPE=[1,1,518,924]


def check_depth(reference, actual):
    import numpy as np
    if (reference.shape!=actual.shape or reference.dtype!=np.float32 or actual.dtype!=np.float32
            or not np.isfinite(reference).all() or not np.isfinite(actual).all()
            or not (reference>0).all() or not (actual>0).all()):
        raise ValueError("Invalid metric depth map")
    delta=np.abs(reference-actual)
    result=dict(passed=bool(np.all(delta<=.001+.001*np.abs(reference))),
                max_abs_delta_m=float(delta.max()),mean_abs_delta_m=float(delta.mean()))
    if not result["passed"]:raise RuntimeError(f"Depth export parity failed: {result}")
    return result


def prepare(a):
    import numpy as np
    import torch
    import onnx
    from PIL import Image,ImageOps
    from transformers import AutoImageProcessor,AutoModelForDepthEstimation
    if a.output.is_symlink():raise ValueError("Use private non-symlink output")
    weights=comparison.verify_weights(a.weights);environment=comparison.runtime()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    identity=dict(revision=REVISION,implementation_sha256=common.file_sha(Path(__file__)),
        source_hf_repo=comparison.REPO,source_hf_revision=comparison.HF_REVISION,
        source_hf_files=weights,environment=environment,input_archive_sha256=INPUT_SHA,
        yolo_bundle_sha256=SPECS["yolo26n"]["bundle"],camera_canvas=[1280,720],
        canvas_policy="Preserve KITTI aspect ratio with grey padding; engineering fixture, not accuracy data",
        input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE)
    common.immutable(a.output/"export_identity.json",common.encoded(identity))
    processor=AutoImageProcessor.from_pretrained(a.weights,local_files_only=True,trust_remote_code=False,use_fast=False)
    native=AutoModelForDepthEstimation.from_pretrained(a.weights,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,attn_implementation="eager").float().cpu().eval()
    class Wrapper(torch.nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,images):return self.model(pixel_values=images).predicted_depth.unsqueeze(1)
    wrapper=Wrapper(native).eval()
    models=a.output/"models";models.mkdir(parents=True,exist_ok=True)
    path=models/"depth_anything_v2_metric_outdoor_small.onnx"
    receipt=path.with_suffix(".export.json")
    with checked_archive(a.inputs,INPUT_SHA) as archive,checked_archive(a.yolo_bundle,SPECS["yolo26n"]["bundle"]) as yolo:
        source=json.loads(yolo.read("input_manifest.json"))
        common.verify_manifest(source,common.load_config())
        if source["manifest_sha256"]!=MANIFEST_SHA or set(archive.namelist())!=set(source["files"]):
            raise ValueError("Reviewed input identity differs")
        for name,digest in source["files"].items():
            if common.sha(archive.read(name))!=digest:raise ValueError("Changed input member")
        det_graph=yolo.read(SPECS["yolo26n"]["member"]);dep_graph=yolo.read("models/yolo26n-depth.onnx")
        if common.sha(det_graph)!=SPECS["yolo26n"]["graph"] or common.sha(dep_graph)!=GRAPH_SHA:
            raise ValueError("Reviewed YOLO graph changed")
        common.immutable(models/"yolo26n.onnx",det_graph)
        common.immutable(models/"yolo26n-depth.onnx",dep_graph)
        det=comparison.ort_session(det_graph,[1,3,640,640],[1,300,6])
        dep=comparison.ort_session(dep_graph,[1,3,768,768],[1,1,768,768])
        ids=source["ids"][:16]
        def canvas(sample):
            original=Image.open(io.BytesIO(archive.read(f"training/image_2/{sample}.png"))).convert("RGB")
            return np.asarray(ImageOps.pad(original,(1280,720),method=Image.Resampling.BILINEAR,color=(114,114,114)))
        example=processor(images=canvas(ids[0]),return_tensors="pt")["pixel_values"]
        if list(example.shape)!=INPUT_SHAPE:raise ValueError("Official 720p processor shape changed")
        if path.exists():
            if not receipt.is_file():raise ValueError("Preserve unreceipted export; use new output")
            r=json.loads(receipt.read_text())
            if r["identity_sha256"]!=common.identity(identity) or r["onnx_sha256"]!=common.file_sha(path):
                raise ValueError("Preserve changed export")
        else:
            print("Exporting unchanged metric Outdoor Small to fixed FP32 ONNX…",flush=True)
            with torch.inference_mode():
                torch.onnx.export(wrapper,(example,),str(path),input_names=["images"],output_names=["output0"],
                    opset_version=17,dynamo=False,do_constant_folding=True,export_params=True)
            common.immutable(receipt,common.encoded(dict(identity_sha256=common.identity(identity),onnx_sha256=common.file_sha(path))))
        graph=onnx.load(str(path),load_external_data=False);onnx.checker.check_model(graph)
        if any(n.domain not in ("","ai.onnx") for n in graph.graph.node) or any(t.data_location==onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise ValueError("Phone graph must be self-contained with standard operators")
        challenger=comparison.ort_session(str(path),INPUT_SHAPE,OUTPUT_SHAPE)
        def tensor(name,value):
            raw=np.ascontiguousarray(value,dtype="<f4").tobytes();common.immutable(a.output/name,raw)
            return dict(file=name,bytes=len(raw),sha256=common.sha(raw),shape=list(value.shape))
        fixtures=[]
        for index,sample in enumerate(ids):
            rgb=canvas(sample);hf=processor(images=rgb,return_tensors="pt")["pixel_values"]
            det_input=letterbox(rgb);dep_input=preprocess(rgb)
            d=det.run(["output0"],{"images":det_input})[0];y=dep.run(["output0"],{"images":dep_input})[0]
            with torch.inference_mode():reference=wrapper(hf).numpy()
            actual=challenger.run(["output0"],{"images":hf.numpy()})[0]
            checked=check_depth(reference,actual)
            checked_y=check_depth(y,y)
            image=io.BytesIO();Image.fromarray(rgb).save(image,format="PNG")
            common.immutable(a.output/f"fixtures/{sample}/image.png",image.getvalue())
            fixtures.append(dict(sample_id=sample,source_rgb_sha256=common.sha(rgb.tobytes()),
                detector_input=tensor(f"fixtures/{sample}/detector_input.bin",det_input),
                detector_reference=tensor(f"fixtures/{sample}/detector_reference.bin",d),
                depth_inputs={"yolo26n_depth":tensor(f"fixtures/{sample}/yolo_input.bin",dep_input),
                              "depth_anything_v2":tensor(f"fixtures/{sample}/dav2_input.bin",hf.numpy())},
                depth_references={"yolo26n_depth":tensor(f"fixtures/{sample}/yolo_reference.bin",y),
                                  "depth_anything_v2":tensor(f"fixtures/{sample}/dav2_reference.bin",actual)},
                native_to_onnx_parity=checked,yolo_positive_finite=checked_y["passed"]))
            print(f"Verified phone fixture {index+1}/16 {sample}: DAV2 max delta {checked['max_abs_delta_m']:.6f}m",flush=True)
        manifest=dict(revision=REVISION,complete=True,onnxruntime_version="1.30.0",models=[
            dict(id="yolo26n_depth",label="YOLO26n-depth",file="models/yolo26n-depth.onnx",sha256=GRAPH_SHA,
                 input_shape=[1,3,768,768],output_shape=[1,1,768,768]),
            dict(id="depth_anything_v2",label="Depth Anything V2 Metric Outdoor Small",file="models/depth_anything_v2_metric_outdoor_small.onnx",
                 sha256=common.file_sha(path),input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE)],
            shared_detector=dict(file="models/yolo26n.onnx",sha256=SPECS["yolo26n"]["graph"],input_shape=[1,3,640,640],output_shape=[1,300,6]),
            fixtures=fixtures,atol_m=.001,rtol=.001,threads=2,spinning=False,
            preflight_order=["yolo26n_depth","depth_anything_v2"],benchmark_order=["yolo26n_depth","depth_anything_v2","depth_anything_v2","yolo26n_depth"],
            warmups=3,timed_predictions=20,sustained_seconds=20,cooldown_min_seconds=15,nominal_continuous_seconds=10,
            host_identity_sha256=common.identity(identity),
            camera_canvas_is_engineering_fixture=True,native_camera_preprocessing_verified=False,
            training_performed=False,model_weights_changed=False,scale_fitting_performed=False,
            surface_accuracy_evaluated=False,live_camera_performance_measured=False,deployment_qualified=False)
        raw=common.encoded(manifest);common.immutable(a.output/"manifest.json",raw)
        print("Phone package manifest SHA256:",common.sha(raw),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ["weights","inputs","yolo-bundle","output"]:p.add_argument("--"+n,type=Path,required=True)
    prepare(p.parse_args())
