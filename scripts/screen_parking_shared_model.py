"""Build/export a real shared-backbone Urban+depth runtime prototype, no training.

CPU-only, synthetic fixed fixtures. Pretrained Urban outputs are preserved before
the new product-only decoder; the depth head is UNTRAINED and cannot measure range.
The official pickled checkpoint is loaded only after its pinned SHA256 and the
clean pinned upstream source have been verified. Never load user-supplied weights.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common

CONFIG = ROOT/"configs/parking_shared_nano_v1.json"


def preprocess(rgb, height=384, width=640, intrinsics=None):
    import cv2
    import numpy as np
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected original RGB uint8 HWC image")
    h,w = rgb.shape[:2]
    gain = min(width/w,height/h)
    nw,nh = round(w*gain),round(h*gain)
    left,top = (width-nw)//2,(height-nh)//2
    canvas = np.full((height,width,3),114,dtype=np.uint8)
    canvas[top:top+nh,left:left+nw] = cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
    affine = np.array([[nw/w,0,left],[0,nh/h,top],[0,0,1]],dtype=np.float64)
    transformed = None
    if intrinsics is not None:
        k = np.asarray(intrinsics,dtype=np.float64)
        if k.shape != (3,3) or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0:
            raise ValueError("Invalid camera intrinsics")
        transformed = (affine@k).tolist()
    tensor = np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32)/255.
    return tensor,dict(original_size=[w,h],input_size=[width,height],resized_size=[nw,nh],
        padding_left_top=[left,top],original_to_input_affine=affine.tolist(),input_intrinsics=transformed)


def fixtures(shape):
    import numpy as np
    h,w = 720,1280
    y,x = np.indices((h,w))
    images = [np.stack([(x*3+y)%256,(x+y*5)%256,(x*7+y*11)%256],axis=-1).astype(np.uint8),
        np.stack([((x//33+y//31)%2)*255,(x*255//w),(y*255//h)],axis=-1).astype(np.uint8),
        np.full((h,w,3),114,np.uint8)]
    return [preprocess(rgb,shape[2],shape[3])[0] for rgb in images]


def decode_candidates(value, metadata, confidence=.25, maximum=100):
    """Explicit new product-only policy; NOT a claim of legacy TopK equivalence."""
    import numpy as np
    if value.ndim != 3 or value.shape[:2] != (1,32) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Invalid raw shared-model candidate output")
    scores = value[0,4:8]
    if ((scores < 0)|(scores > 1)).any():raise ValueError("Invalid class probabilities")
    selected_class = scores.argmax(axis=0)
    probability = scores.max(axis=0)
    eligible = np.flatnonzero(probability > confidence)
    order = eligible[np.argsort(-probability[eligible],kind="stable")][:maximum]
    inverse = np.linalg.inv(np.asarray(metadata["original_to_input_affine"]))
    width,height = metadata["original_size"]
    result = []
    for index in order:
        def original(x,y):
            a = inverse@np.array([x,y,1.])
            return [float(np.clip(a[0],0,width)),float(np.clip(a[1],0,height))]
        row = value[0,:,index]
        box = original(*row[:2])+original(*row[2:4])
        corners = row[8:].reshape(8,3)
        result.append(dict(class_name="Pedestrian" if selected_class[index] == 0 else "Vehicle",
            coco_class_id=[0,2,5,7][selected_class[index]],score=float(probability[index]),bbox=box,
            projected_corners_xy_confidence=[original(*corner[:2])+[float(corner[2])] for corner in corners],
            distance_m=None,depth_head_untrained=True))
    return result


def state_digest(module):
    """Tensor-value identity, independent of pickle/zip metadata."""
    import hashlib
    digest = hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(common.encoded(dict(name=name,shape=list(array.shape),dtype=str(array.dtype))))
        digest.update(array.tobytes())
    return digest.hexdigest()


def compare(reference, actual, atol, rtol):
    import numpy as np
    if reference.shape != actual.shape or reference.dtype != np.float32 or actual.dtype != np.float32:
        raise ValueError("Graph shape/dtype changed")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():raise ValueError("Non-finite outputs")
    difference = np.abs(reference-actual)
    result = dict(passed=bool(np.all(difference <= atol+rtol*np.abs(reference))),
        max_abs_delta=float(difference.max()),mean_abs_delta=float(difference.mean()))
    if not result["passed"]:raise RuntimeError(f"Strict tensor export parity failed: {result}")
    return result


def publish_bundle(output):
    """Stable archive metadata makes successful reruns verifiable, not destructive."""
    buffer=io.BytesIO()
    members={"run_identity.json","shared_model_screen.json","phone_runtime_manifest.json",
        "models/parking_shared_nano.onnx","models/parking_shared_nano.export.json"}
    phone_path=output/"phone_runtime_manifest.json"
    if phone_path.is_file():
        for row in json.loads(phone_path.read_text())["fixtures"]:
            members.update(spec["file"] for spec in [row["input"],*row["references"].values()])
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
        for member in sorted(members):
            path=output/member
            if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("Unsafe runtime package member")
            if path.is_file():
                item=zipfile.ZipInfo(member,date_time=(1980,1,1,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(item,path.read_bytes())
    destination=output/"parking_shared_runtime_results.zip"
    common.immutable(destination,buffer.getvalue())
    common.immutable(output/"bundle_receipt.json",common.encoded(dict(file=destination.name,sha256=common.file_sha(destination))))


def reopen(output, run):
    report_path=output/"shared_model_screen.json";phone_path=output/"phone_runtime_manifest.json"
    if not (report_path.is_file() and phone_path.is_file()):return None
    report=json.loads(report_path.read_text());phone=json.loads(phone_path.read_text())
    if report["run_identity"] != common.identity(run) or not report["complete"]:
        raise ValueError("Preserve changed completed screen; use a new RUN_ID")
    if common.file_sha(output/report["onnx_file"]) != report["onnx_sha256"] or phone["model_sha256"] != report["onnx_sha256"]:
        raise ValueError("Saved graph changed")
    for row in phone["fixtures"]:
        for spec in [row["input"],*row["references"].values()]:
            path=output/spec["file"]
            if not path.resolve().is_relative_to(output.resolve()) or path.is_symlink() or common.file_sha(path) != spec["sha256"]:
                raise ValueError("Saved runtime fixture changed")
    publish_bundle(output)
    return report


def screen(a):
    # Set before importing the upstream package; do not reset global settings.
    settings = a.output/"runtime_settings";settings.mkdir(parents=True,exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings.resolve())
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from models.parking_shared import SharedParkingNet, REVISION, OUTPUT_NAMES, metric_depth_loss
    from scripts.screen_parking_edge_candidates import urban_source,download
    config = json.loads(a.config.read_text())
    if config["revision"] != REVISION:raise ValueError("Model and screen revisions differ")
    if a.output.is_symlink() or any(a.output.resolve().is_relative_to(p.resolve()) for p in [a.source,a.weights]):
        raise ValueError("Keep separate non-symlink run/source/checkpoint directories")
    versions = {k:importlib.metadata.version(k) for k in ["ultralytics","torch","torchvision","numpy","pillow"]}
    if (versions["ultralytics"] != config["urban_ultralytics_version"] or onnx.__version__ != config["onnx_version"]
            or ort.__version__ != config["onnxruntime_version"]):
        raise ValueError("Use the notebook's isolated/pinned prototype dependencies")
    source_settings = dict(source_url=config["urban_source_url"],source_commit=config["urban_source_commit"])
    source_commit = urban_source(a.source,source_settings)
    checkpoint = download(config["urban_checkpoint"],a.weights)
    print("Verified official Urban checkpoint SHA256 before upstream object loading",flush=True)
    from ultralytics import YOLO
    torch.manual_seed(20261027);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    native = YOLO(str(checkpoint),task="pose").model.float().cpu().eval()
    native.fuse(verbose=False)
    head = native.model[-1];head.export=True;head.format="onnx";head.dynamic=False
    reference = copy.deepcopy(native).eval()
    model = SharedParkingNet(native,config["shared_features"]["channels"],config["depth_decoder_width"]).eval()
    original_body_sha = state_digest(model.body)
    implementation = [Path(__file__),ROOT/"models/parking_shared.py",ROOT/"scripts/parking_edge_common.py",
        ROOT/"scripts/screen_parking_edge_candidates.py"]
    run = dict(revision=REVISION,config=config,source_commit=source_commit,
        checkpoint_sha256=common.file_sha(checkpoint),source_body_state_sha256=original_body_sha,
        implementation_sha256={p.name:common.file_sha(p) for p in implementation},
        runtime=dict(python=platform.python_version(),platform=platform.platform(),packages=dict(versions,onnx=onnx.__version__,onnxruntime=ort.__version__)),
        fixtures="three deterministic synthetic RGB patterns, NOT validation/training images",
        model_initial_state_sha256=state_digest(model),training_performed=False,optimizer_steps=0)
    common.immutable(a.output/"run_identity.json",common.encoded(run))
    existing=reopen(a.output,run)
    if existing is not None:
        print("Verified completed runtime screen; no inference or training repeated",flush=True)
        print(json.dumps(existing,indent=2));return
    inputs = fixtures(config["input_shape"])
    initial_decoder_state=copy.deepcopy(model.depth_decoder.state_dict())
    checks = [];references = []
    for index,tensor in enumerate(inputs):
        image = torch.from_numpy(tensor)
        with torch.inference_mode():
            source_output = reference(image)
            candidates,depth = model(image)
            features = model.image_features(image)
            all_candidates = model.full_urban_candidates(features)
            restored_source = model.body[-1].postprocess(all_candidates.permute(0,2,1))
        preserved = compare(source_output.numpy(),restored_source.numpy(),0.,0.)
        if depth.shape != (1,1,config["input_shape"][2]//4,config["input_shape"][3]//4) or not (depth > 0).all():
            raise ValueError("Prototype depth output shape/positivity failed")
        checks.append(dict(fixture=index,pretrained_native_output_preservation=preserved))
        references.append((candidates.numpy(),depth.numpy()))
    # Validate that the new decoder has finite gradients and the frozen body,
    # including all BN running statistics, remains bit-identical. No optimizer.
    model.train();_,depth = model(torch.from_numpy(inputs[0]))
    loss = metric_depth_loss(depth,torch.full_like(depth,3.),torch.ones_like(depth,dtype=torch.bool))
    loss.backward()
    grads = [p.grad for p in model.depth_decoder.parameters()]
    finite = all(g is not None and torch.isfinite(g).all() for g in grads)
    nonzero = sum(int(torch.count_nonzero(g)) for g in grads if g is not None)
    if not finite or not nonzero or any(p.grad is not None for p in model.body.parameters()) or state_digest(model.body) != original_body_sha:
        raise ValueError("Gradient/frozen-body protection smoke failed")
    model.zero_grad(set_to_none=True)
    # Discard synthetic BN updates in the new head; restore its exact initial
    # tensors instead of relying on RNG state consumed by upstream model fusion.
    model.depth_decoder.load_state_dict(initial_decoder_state,strict=True)
    model.eval()
    if state_digest(model) != run["model_initial_state_sha256"]:
        raise ValueError("Synthetic gradient check did not restore the exact export initialization")
    graph = a.output/"models/parking_shared_nano.onnx";receipt=graph.with_suffix(".export.json")
    if graph.exists():
        if not receipt.is_file() or json.loads(receipt.read_text()) != dict(run_identity=common.identity(run),onnx_sha256=common.file_sha(graph)):
            raise ValueError("Preserve unreceipted/changed export; choose a new RUN_ID")
    else:
        graph.parent.mkdir(parents=True,exist_ok=True)
        with torch.inference_mode():
            torch.onnx.export(model,(torch.from_numpy(inputs[0]),),str(graph),input_names=["image"],
                output_names=list(OUTPUT_NAMES),opset_version=config["onnx_opset"],dynamo=False,
                do_constant_folding=True,export_params=True)
        common.immutable(receipt,common.encoded(dict(run_identity=common.identity(run),onnx_sha256=common.file_sha(graph))))
    loaded = onnx.load(str(graph),load_external_data=False);onnx.checker.check_model(loaded)
    if any(n.domain not in ("","ai.onnx") for n in loaded.graph.node) or any(t.data_location == onnx.TensorProto.EXTERNAL for t in loaded.graph.initializer):
        raise ValueError("Use self-contained standard ONNX operators only")
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.add_session_config_entry("session.intra_op.allow_spinning","0")
    options.add_session_config_entry("session.inter_op.allow_spinning","0")
    session=ort.InferenceSession(str(graph),options,providers=["CPUExecutionProvider"])
    expected_outputs=[("candidates",[1,32,5040],"tensor(float)"),("depth_untrained",[1,1,96,160],"tensor(float)")]
    if (session.get_providers()!=["CPUExecutionProvider"]
            or [(x.name,x.shape,x.type) for x in session.get_inputs()] != [("image",config["input_shape"],"tensor(float)")]
            or [(x.name,x.shape,x.type) for x in session.get_outputs()] != expected_outputs):
        raise ValueError("Prototype ONNX interface differs from the documented phone contract")
    fixture_specs=[]
    for index,(tensor,reference_values) in enumerate(zip(inputs,references)):
        actual=session.run(list(OUTPUT_NAMES),{"image":tensor})
        checks[index]["onnx_candidate_parity"]=compare(reference_values[0],actual[0],config["parity"]["candidate_atol"],config["parity"]["candidate_rtol"])
        checks[index]["onnx_depth_parity"]=compare(reference_values[1],actual[1],config["parity"]["depth_atol"],config["parity"]["depth_rtol"])
        def store(name,value):
            raw=np.ascontiguousarray(value,dtype="<f4").tobytes();common.immutable(a.output/name,raw)
            return dict(file=name,shape=list(value.shape),bytes=len(raw),sha256=common.sha(raw))
        fixture_specs.append(dict(id=f"synthetic_{index}",input=store(f"fixtures/{index}/image.bin",tensor),
            references={name:store(f"fixtures/{index}/{name}.bin",value) for name,value in zip(OUTPUT_NAMES,reference_values)}))
    for _ in range(config["warmups"]):session.run(None,{"image":inputs[0]})
    durations=[]
    for index in range(config["timed_runs"]):
        start=time.perf_counter();session.run(None,{"image":inputs[index%len(inputs)]});durations.append((time.perf_counter()-start)*1000)
    total=sum(p.numel() for p in model.parameters());decoder=sum(p.numel() for p in model.depth_decoder.parameters())
    report=dict(complete=True,revision=REVISION,run_identity=common.identity(run),onnx_file="models/parking_shared_nano.onnx",
        onnx_sha256=common.file_sha(graph),onnx_bytes=graph.stat().st_size,input_shape=config["input_shape"],
        outputs=[dict(name=n,shape=s,dtype=t) for n,s,t in expected_outputs],parameters=dict(total=total,frozen_urban=total-decoder,new_depth_decoder=decoder),
        shared_backbone_and_neck_passes_per_image=1,checks=checks,gradient_smoke=dict(finite=True,nonzero_gradient_values=nonzero,
            pretrained_weights_and_BN_unchanged=True,synthetic_decoder_BN_updates_discarded=True,optimizer_steps=0),
        host_onnx_cpu_profile=dict(samples=len(durations),median_ms=float(np.median(durations)),p95_ms=float(np.quantile(durations,.95)),
            scope="ONNX session.run, CPU-2, includes fetching both outputs; excludes RGB preprocessing, candidate decoding, depth restoration, camera and rendering; NOT iPhone latency"),
        output_bytes_per_image=sum(int(np.prod(s))*4 for _,s,_ in expected_outputs),
        depth_head_untrained=True,depth_head_usable_for_measurement=False,source_Urban_pretrained=True,
        product_selection_is_new_policy=True,accuracy_evaluated=False,metric_3d_evaluated=False,
        training_performed=False,optimizer_steps=0,held_out_dataset_images_used=0,iphone_performance_measured=False,
        phone_app_changed=False,deployment_qualified=False,next_step="Camera-OFF iPhone runtime adapter and benchmark before GT depth training")
    common.immutable(a.output/"shared_model_screen.json",common.encoded(report))
    phone=dict(revision=REVISION,complete=True,model_file=report["onnx_file"],model_sha256=report["onnx_sha256"],
        input_name="image",output_names=list(OUTPUT_NAMES),input_shape=config["input_shape"],outputs=report["outputs"],fixtures=fixture_specs,
        provider="CPUExecutionProvider",intra_op_threads=2,inter_op_threads=1,spinning=False,
        permitted_use="camera-OFF synthetic runtime test ONLY; no distance labels or live camera mode",
        depth_head_untrained=True,accuracy_evaluated=False,deployment_qualified=False)
    common.immutable(a.output/"phone_runtime_manifest.json",common.encoded(phone))
    publish_bundle(a.output)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=CONFIG)
    for name in ["output","source","weights"]:p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    try:
        screen(args)
    except Exception as error:
        import traceback
        from datetime import datetime,timezone
        failure=dict(complete=False,stage="shared_runtime_screen",error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc(),training_performed=False,optimizer_steps=0,
            phone_app_changed=False,depth_head_usable_for_measurement=False)
        if not args.output.is_symlink():
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            try:common.immutable(args.output/f"errors/failure_{stamp}.json",common.encoded(failure))
            except OSError:pass
        raise


if __name__ == "__main__":main()
