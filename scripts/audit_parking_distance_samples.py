"""Validate opt-in clean parking samples and inspect fixed depth ROIs.

No fitting or model changes. Optional CPU replay checks native-output numeric
preservation, not whether the model's nominal meters match a tape reference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import parking_edge_common as common
from scripts.prepare_parking_depth_phone import GRAPH_SHA, preprocess, parity

REVISION = "PARKING-CLEAN-DISTANCE-SAMPLE-AUDIT-2026-10-25-r1"


def checked_blob(root, item):
    name=item["file"]
    relative=Path(name)
    if (relative.is_absolute() or ".." in relative.parts or not name.startswith("distance_diagnostics/")
            or "\\" in name or len(relative.parts)!=2):
        raise ValueError("Unsafe diagnostic blob path")
    path=root/relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Symlinked diagnostic blob")
    raw=path.read_bytes()
    if len(raw)!=item["bytes"] or common.sha(raw)!=item["sha256"]:
        raise ValueError("Diagnostic blob bytes/hash differ")
    return raw


def restore_map(depth, height=720, width=1280):
    import numpy as np
    if depth.shape!=(768,768) or depth.dtype!=np.float32 or not np.isfinite(depth).all() or not (depth>0).all():
        raise ValueError("Invalid full depth map")
    x=np.arange(width,dtype=np.float32)*(np.float32(767)/np.float32(width-1))
    y=np.arange(height,dtype=np.float32)*(np.float32(767)/np.float32(height-1))
    x0=x.astype(np.int32); y0=y.astype(np.int32)
    x1=np.minimum(x0+1,767); y1=np.minimum(y0+1,767)
    fx=x-x0.astype(np.float32); fy=(y-y0.astype(np.float32))[:,None]
    upper=(1-fx)*depth[y0[:,None],x0[None,:]]+fx*depth[y0[:,None],x1[None,:]]
    lower=(1-fx)*depth[y1[:,None],x0[None,:]]+fx*depth[y1[:,None],x1[None,:]]
    return (1-fy)*upper+fy*lower


def region_stats(depth, box, fractions):
    import numpy as np
    if len(box)!=4 or not all(math.isfinite(x) for x in box) or box[2]<=box[0] or box[3]<=box[1]:
        raise ValueError("Invalid diagnostic box")
    x1,y1,x2,y2=box; a,b,c,d=fractions
    left=max(0,math.floor(x1+(x2-x1)*a)); right=min(depth.shape[1],math.ceil(x1+(x2-x1)*c))
    top=max(0,math.floor(y1+(y2-y1)*b)); bottom=min(depth.shape[0],math.ceil(y1+(y2-y1)*d))
    if right<=left or bottom<=top:
        raise ValueError("Empty diagnostic region")
    values=depth[top:bottom,left:right]
    return dict(roi_xyxy=[left,top,right,bottom],p10_m=float(np.quantile(values,.1)),
                distance_m=float(np.median(values)),p90_m=float(np.quantile(values,.9)))


def analyze(run_folder, depth_model=None):
    import io
    import numpy as np
    from PIL import Image
    root=Path(run_folder).resolve()
    manifest_path=root/"distance_diagnostics/manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Symlinked sample manifest")
    manifest=json.loads(manifest_path.read_text())
    if (manifest["revision"]!="PARKING-DISTANCE-DIAGNOSTIC-2026-10-25-r1"
            or manifest["raw_image_has_annotations"] is not False or manifest["depth_model_sha256"]!=GRAPH_SHA
            or manifest["sample_count"]!=len(manifest["samples"]) or not 1<=manifest["sample_count"]<=3):
        raise ValueError("Unsupported or incomplete diagnostic contract")
    report=None
    if not manifest["synthetic"]:
        report=json.loads((root/"report.json").read_text())
        summary=report["distance_diagnostics"]
        if (report["depth_model_sha256"]!=GRAPH_SHA or not report["distance_diagnostic_enabled"]
                or summary["manifest_sha256"]!=common.file_sha(manifest_path)):
            raise ValueError("Run report and diagnostic manifest differ")
    session=None
    if depth_model is not None:
        if manifest["synthetic"]:
            raise ValueError("Synthetic copy-test maps are not native inference references")
        if common.file_sha(depth_model)!=GRAPH_SHA:
            raise ValueError("Replay depth model identity changed")
        import onnxruntime as ort
        if ort.__version__!="1.30.0":
            raise ValueError("Replay requires reviewed ONNX Runtime 1.30.0")
        options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
        for scope in ["intra_op","inter_op"]:
            options.add_session_config_entry(f"session.{scope}.allow_spinning","0")
        session=ort.InferenceSession(str(depth_model),options,providers=["CPUExecutionProvider"])
        if session.get_providers()!=["CPUExecutionProvider"]:
            raise ValueError("Expected CPU-only replay")
    samples=[]; ids=set(); last_time=None; binary_bytes=0
    frames={f["frame_id"]:f for f in report["frames"]} if report else {}
    tags={t["id"]:t for t in report["distance_tags"]} if report else {}
    for sample in manifest["samples"]:
        fid=sample["frame_id"]; capture=sample["capture_host_seconds"]
        if (not isinstance(fid,int) or isinstance(fid,bool) or fid<0 or fid in ids
                or fid!=sample["urban_frame_id"] or fid!=sample["depth_frame_id"]
                or not math.isfinite(capture) or (last_time is not None and capture-last_time<5)):
            raise ValueError("Invalid diagnostic frame/time identity or spacing")
        ids.add(fid);last_time=capture
        if sample["width"]!=1280 or sample["height"]!=720 or not 5120<=sample["row_bytes"]<=8192 or sample["row_bytes"]%4:
            raise ValueError("Invalid raw image layout")
        bgra=checked_blob(root,sample["bgra"]); depth_raw=checked_blob(root,sample["depth"]); png=checked_blob(root,sample["png"])
        binary_bytes+=len(bgra)+len(depth_raw)
        if binary_bytes>32_000_000 or len(bgra)!=sample["row_bytes"]*720 or len(depth_raw)!=768*768*4 or sample["depth_shape"]!=[1,1,768,768]:
            raise ValueError("Diagnostic size/shape/memory contract differs")
        pixels=np.frombuffer(bgra,dtype=np.uint8).reshape(720,sample["row_bytes"])[:,:1280*4].reshape(720,1280,4)
        rgb=pixels[:,:,[2,1,0]]
        with Image.open(io.BytesIO(png)) as clean_image:
            if clean_image.size!=(1280,720):
                raise ValueError("Clean PNG dimensions differ from raw capture")
            view=np.asarray(clean_image.convert("RGB"))
        if not np.array_equal(rgb,view):
            raise ValueError("Clean PNG differs from captured BGRA pixels")
        depth=np.frombuffer(depth_raw,dtype="<f4").reshape(768,768)
        original=restore_map(depth)
        detections=[]
        if report:
            frame=frames.get(fid);tag=tags.get(sample["reference_tag_id"])
            if (frame is None or tag is None or frame["reference_tag_id"]!=tag["id"]
                    or sample["reference"]!=tag["reference"] or capture<tag["start_host_seconds"]+5
                    or capture>=tag["end_host_seconds"]):
                raise ValueError("Diagnostic reference/frame association differs")
            for key in ["capture_host_seconds","source_pts_value","source_pts_timescale","intrinsics_row_major","urban_frame_id","depth_frame_id"]:
                if sample[key]!=frame[key]:
                    raise ValueError(f"Diagnostic frame metadata differs: {key}")
            for detection in frame["detections"]:
                box=detection["bbox"]
                baseline=region_stats(original,box,(.25,.25,.75,.75))
                recorded=detection["range"]
                if baseline["roi_xyxy"]!=recorded["roi_xyxy"] or any(
                        abs(baseline[k]-recorded[k])>.001+.001*abs(recorded[k]) for k in ["distance_m","p10_m","p90_m"]):
                    raise ValueError("Saved full map does not reproduce native ROI values")
                regions={name:region_stats(original,box,fraction) for name,fraction in {
                    "central_half":(.25,.25,.75,.75), "central_quarter":(.375,.375,.625,.625),
                    "lower_central_body":(.25,.5,.75,.85), "upper_central":(.25,.15,.75,.45),
                    "narrow_centre_strip":(.4,.25,.6,.75)}.items()}
                detections.append(dict(class_name=detection["class_name"],bbox=box,score=detection["score"],
                                       native_roi_reproduced=True,exploratory_regions=regions))
        replay=None
        if session:
            value=session.run(["output0"],{"images":preprocess(rgb)})[0]
            replay=parity(depth.reshape(1,1,768,768),value)
            if not replay["passed"]:
                raise ValueError(f"Clean image replay differs from native full depth output at frame {fid}")
        samples.append(dict(frame_id=fid,capture_host_seconds=capture,reference_tag_id=sample["reference_tag_id"],
                            reference=sample["reference"],bgra_png_exact_rgb_match=True,
                            full_depth_positive_finite=True,native_cpu_replay=replay,detections=detections))
    return dict(revision=REVISION,complete=True,samples=samples,sample_count=len(samples),
                synthetic=manifest["synthetic"],manifest_sha256=common.file_sha(manifest_path),
                native_replay_performed=session is not None,model_weights_changed=False,training_performed=False,
                roi_definitions_fixed_before_reference_scoring=True,calibration_fitted=False,correction_applied=False,
                automatic_detection_reference_matching=False,accuracy_evaluated=False,deployment_qualified=False,
                caveat="Alternative ROI values are diagnostic; no region is selected using tape error, and no nearest-surface or sensor-frame range is claimed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder",type=Path,required=True)
    parser.add_argument("--replay-depth-model",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.run_folder,args.replay_depth_model)
    common.immutable(args.output,common.encoded(result))
    print(json.dumps(dict(complete=result["complete"],samples=result["sample_count"],
                         native_replay_performed=result["native_replay_performed"],accuracy_evaluated=False),indent=2))


if __name__=="__main__":
    main()
