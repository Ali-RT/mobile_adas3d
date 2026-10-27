"""Freeze a GT-only nuScenes mini validation depth screen; no model inference.

Uses single keyframe LiDAR sweeps, calibrated ego/sensor poses and lidarseg.
Sparse projection is not dense camera ground truth: parallax/occlusion,
rolling LiDAR acquisition, sensor timing and moving objects remain limitations.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import io
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import parking_edge_common as common

CONFIG=common.ROOT/"configs/parking_lidar_depth_v1.json"
TABLES=["scene","sample","sample_data","calibrated_sensor","ego_pose","sensor","category","instance","sample_annotation","attribute","lidarseg"]


def safe_file(root,name):
    name=Path(name)
    if name.is_absolute() or ".." in name.parts or "\\" in str(name): raise ValueError("Unsafe dataset member")
    path=root/name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()): raise ValueError("Dataset symlink escapes root")
    if not path.is_file():raise FileNotFoundError(path)
    return path


def rotation(wxyz):
    import numpy as np
    from scipy.spatial.transform import Rotation
    q=np.asarray(wxyz,dtype=np.float64)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.001: raise ValueError("Invalid quaternion")
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


def camera_projection(xyz,lidar_sensor,lidar_pose,camera_pose,camera_sensor,width,height):
    import numpy as np
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError("Invalid LiDAR points")
    points=xyz.T.astype(np.float32,copy=True)
    # Keep the official SDK's four transforms and Float32 point storage.
    for record in [lidar_sensor,lidar_pose]:
        points[:]=rotation(record["rotation"])@points
        points[:]+=np.asarray(record["translation"])[:,None]
    world=points.copy()
    for record in [camera_pose,camera_sensor]:
        points[:]-=np.asarray(record["translation"])[:,None]
        points[:]=rotation(record["rotation"]).T@points
    k=np.asarray(camera_sensor["camera_intrinsic"],dtype=np.float64)
    if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0:raise ValueError("Invalid camera matrix")
    projected=k@points
    with np.errstate(divide="ignore",invalid="ignore"): uv=projected[:2]/projected[2]
    z=points[2].copy()
    keep=(z>1)&(uv[0]>1)&(uv[0]<width-1)&(uv[1]>1)&(uv[1]<height-1)&np.isfinite(uv).all(axis=0)
    return uv[:,keep].T,z[keep],np.flatnonzero(keep),world


def rasterize(uv,z,labels,static,width,height,config):
    import numpy as np
    from scipy.ndimage import minimum_filter,maximum_filter
    if (uv.shape!=(len(z),2) or labels.shape!=z.shape or static.shape!=z.shape
            or not np.isfinite(uv).all() or not np.isfinite(z).all() or not (z>0).all()):
        raise ValueError("Mismatched/invalid projected arrays")
    xy=np.rint(uv).astype(np.int32)
    keep=(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
    xy,z,labels,static=xy[keep],z[keep],labels[keep],static[keep]
    pixel=xy[:,1]*width+xy[:,0]
    order=np.lexsort((np.arange(len(z)),z,pixel))
    first=np.r_[True,pixel[order][1:]!=pixel[order][:-1]] if len(order) else np.zeros(0,dtype=bool)
    selected=order[first]
    xy,z,labels,static=xy[selected],z[selected],labels[selected],static[selected]
    # Nearest point wins BEFORE dropping noise/ego labels. Do not reveal a
    # farther hidden return by removing a closer invalid semantic point.
    depth_lo=np.full((height,width),np.inf,dtype=np.float32)
    depth_hi=np.full((height,width),-np.inf,dtype=np.float32)
    depth_lo[xy[:,1],xy[:,0]]=z;depth_hi[xy[:,1],xy[:,0]]=z
    guard=config["gt_boundary_guard"];window=guard["window_pixels"]
    lo=minimum_filter(depth_lo,size=window,mode="constant",cval=np.inf)
    hi=maximum_filter(depth_hi,size=window,mode="constant",cval=-np.inf)
    conservative=(hi[xy[:,1],xy[:,0]]-lo[xy[:,1],xy[:,0]])<=guard["maximum_local_depth_span_m"]
    valid=(z>=config["depth_min_m"])&(z<=config["depth_max_m"])&~np.isin(labels,config["invalid_semantic_indices"])
    return dict(xy=xy[valid],z=z[valid].astype(np.float32),semantic=labels[valid].astype(np.uint8),
                static=static[valid].astype(bool),conservative=conservative[valid],
                raster_points_before_range_semantic_filter=len(z))


def cell_counts(points,mapping,config):
    import numpy as np
    result={}
    for name,indices in mapping.items():
        cls=np.isin(points["semantic"],indices)&points["conservative"]
        for low,high in config["range_bands_m"]:
            mask=cls&(points["z"]>=low)&(points["z"]<(high if high!=80 else np.nextafter(80.,np.inf)))
            result[f"{name}:{low}-{high}"]=int(mask.sum())
        result[f"{name}:2-6"]=int((cls&(points["z"]>=2)&(points["z"]<=6)).sum())
    return result


def select(rows,config):
    cells=[f"{cls}:{low}-{high}" for cls in ["Vehicle","Pedestrian"] for low,high in config["range_bands_m"][:4]]
    rank=lambda r:(common.sha((config["selection_seed"]+r["camera_token"]).encode()),r["camera_token"])
    ranked=sorted(rows,key=rank);selected=[];used_samples=set()
    while len(selected)<config["maximum_images"]:
        changed=False
        for cell in cells:
            candidate=next((r for r in ranked if r["sample_token"] not in used_samples
                and r["counts"][cell]>=config["minimum_valid_points_for_selection_cell"]),None)
            if candidate:
                selected.append(candidate);used_samples.add(candidate["sample_token"]);changed=True
                if len(selected)==config["maximum_images"]:break
        if not changed:break
    for r in ranked:
        if len(selected)>=config["maximum_images"]:break
        if r["sample_token"] not in used_samples:
            selected.append(r);used_samples.add(r["sample_token"])
    return selected


def prepare(a):
    import numpy as np
    import importlib.metadata
    import nuscenes
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import mini_val
    from nuscenes.utils.geometry_utils import points_in_box
    from pyquaternion import Quaternion
    config=json.loads(a.config.read_text())
    if sorted(mini_val)!=sorted(config["scenes"]):raise ValueError("Expected official mini_val split")
    root=a.dataset.resolve();output=a.output.resolve()
    if output.is_relative_to(root) or output.is_symlink():raise ValueError("Keep output separate from read-only dataset")
    tables={name:common.file_sha(safe_file(root,f"v1.0-mini/{name}.json")) for name in TABLES}
    nusc=NuScenes(version="v1.0-mini",dataroot=str(root),verbose=False)
    mapping={name:[nusc.lidarseg_name2idx_mapping[c] for c in names] for name,names in config["classes"].items()}
    identity=dict(revision=config["revision"],config_sha256=common.file_sha(a.config),dataset_root=str(root),
        table_sha256=tables,implementation_sha256=common.file_sha(Path(__file__)),
        sdk_version=importlib.metadata.version("nuscenes-devkit"),
        sdk_source_sha256=common.file_sha(Path(nuscenes.__file__).parent/"nuscenes.py"),mapping=mapping)
    common.immutable(output/"prepare_identity.json",common.encoded(identity))
    samples=sorted([s for s in nusc.sample if nusc.get("scene",s["scene_token"])["name"] in config["scenes"]],key=lambda s:s["token"])
    rows=[];clouds={};points_cache={};excluded=Counter();checks=[];checked_channels=set()
    for si,sample in enumerate(samples):
        lidar=nusc.get("sample_data",sample["data"]["LIDAR_TOP"])
        lp=nusc.get("ego_pose",lidar["ego_pose_token"]);ls=nusc.get("calibrated_sensor",lidar["calibrated_sensor_token"])
        label=nusc.get("lidarseg",lidar["token"])
        cloud_path=safe_file(root,lidar["filename"]);label_path=safe_file(root,label["filename"])
        raw=np.fromfile(cloud_path,dtype="<f4")
        if raw.size%5:raise ValueError("Malformed five-channel nuScenes LiDAR")
        xyz=raw.reshape(-1,5)[:,:3].copy();sem=np.fromfile(label_path,dtype=np.uint8)
        if len(sem)!=len(xyz) or len(xyz)==0 or sem.max()>31:raise ValueError("LiDAR/semantic point count differs")
        world=rotation(lp["rotation"])@(rotation(ls["rotation"])@xyz.T+np.asarray(ls["translation"])[:,None])+np.asarray(lp["translation"])[:,None]
        static=np.zeros(len(xyz),dtype=bool)
        for token in sample["anns"]:
            annotation=nusc.get("sample_annotation",token)
            attributes={nusc.get("attribute",x)["name"] for x in annotation["attribute_tokens"]}
            cls=next((k for k,v in config["classes"].items() if annotation["category_name"] in v),None)
            if cls and attributes.intersection(config["static_attributes"]):
                static|=points_in_box(nusc.get_box(token),world)&np.isin(sem,mapping[cls])
        clouds[lidar["token"]]=dict(cloud_file=lidar["filename"],cloud_sha256=common.file_sha(cloud_path),
            labels_file=label["filename"],labels_sha256=common.file_sha(label_path))
        for channel in config["camera_channels"]:
            camera=nusc.get("sample_data",sample["data"][channel])
            dt=abs(camera["timestamp"]-lidar["timestamp"])/1000
            if dt>config["max_camera_lidar_timestamp_difference_ms"]:
                excluded["timestamp_gap_over_limit"]+=1;continue
            cp=nusc.get("ego_pose",camera["ego_pose_token"]);cs=nusc.get("calibrated_sensor",camera["calibrated_sensor_token"])
            safe_file(root,camera["filename"])
            uv,z,indices,_=camera_projection(xyz,ls,lp,cp,cs,camera["width"],camera["height"])
            if channel not in checked_channels:
                sdk_uv,sdk_z,image=nusc.explorer.map_pointcloud_to_image(lidar["token"],camera["token"])
                if sdk_uv[:2].T.shape!=uv.shape or sdk_z.shape!=z.shape:raise ValueError("Official SDK projection mask differs")
                duv=float(np.abs(sdk_uv[:2].T-uv).max(initial=0));dz=float(np.abs(sdk_z-z).max(initial=0))
                if duv>.002 or dz>.0002:raise ValueError("Official SDK projection parity failed")
                checks.append(dict(channel=channel,uv_max_abs_px=duv,z_max_abs_m=dz,passed=True));checked_channels.add(channel)
            points=rasterize(uv,z,sem[indices],static[indices],camera["width"],camera["height"],config)
            pose_delta=float(np.linalg.norm(np.asarray(cp["translation"])-lp["translation"]))
            points["static"]&=pose_delta<=config["max_ego_translation_between_sensor_poses_for_static_subset_m"]
            if len(points["z"])==0:excluded["no_valid_projected_points"]+=1;continue
            row=dict(camera_token=camera["token"],lidar_token=lidar["token"],sample_token=sample["token"],
                scene=nusc.get("scene",sample["scene_token"])["name"],channel=channel,
                timestamp_us=camera["timestamp"],camera_lidar_timestamp_difference_ms=dt,
                ego_translation_between_sensor_poses_m=pose_delta,image_file=camera["filename"],
                width=camera["width"],height=camera["height"],intrinsics=cs["camera_intrinsic"],counts=cell_counts(points,mapping,config))
            rows.append(row);points_cache[camera["token"]]=points
        if (si+1)%10==0:print(f"Indexed {si+1}/{len(samples)} validation keyframes; {len(rows)} eligible camera views",flush=True)
    selected=select(rows,config)
    if not selected:raise ValueError("No validated image/LiDAR pairs; no model inference can start")
    for row in selected:
        row["image_sha256"]=common.file_sha(safe_file(root,row["image_file"]))
        row.update(clouds[row["lidar_token"]])
        data=points_cache[row["camera_token"]];buffer=io.BytesIO()
        np.savez_compressed(buffer,**{k:v for k,v in data.items() if k!="raster_points_before_range_semantic_filter"})
        member=f"ground_truth/{row['camera_token']}.npz";common.immutable(output/member,buffer.getvalue())
        row["ground_truth_file"]=member;row["ground_truth_sha256"]=common.sha(buffer.getvalue())
    inventory=dict(complete=True,available_validation_keyframes=len(samples),eligible_camera_images=len(rows),
        excluded=dict(excluded),official_sdk_projection_checks=checks,
        available_frames_per_cell={k:sum(r["counts"][k]>=config["minimum_valid_points_for_selection_cell"] for r in rows) for k in rows[0]["counts"]},
        selected_images=len(selected),selected_unique_keyframes=len({r["sample_token"] for r in selected}),
        selected_scenes=dict(Counter(r["scene"] for r in selected)),
        selected_frames_per_cell={k:sum(r["counts"][k]>=config["minimum_valid_points_for_selection_cell"] for r in selected) for k in rows[0]["counts"]})
    manifest=dict(schema_version=1,complete=True,revision=config["revision"],dataset_root=str(root),
        prepare_identity_sha256=common.identity(identity),config=config,mapping=mapping,samples=selected,
        inventory=inventory,GT_source="Single-sweep LiDAR transformed at the two sensor poses; camera-axis Z; lidarseg point classes",
        limitations=["Two mini_val scenes; enriched selection, not representative full validation",
            "Nearest-per-pixel z-buffer and GT-only depth-boundary guard do not solve all cross-sensor occlusion",
            "Moving objects and rolling LiDAR acquisition are not deskewed; conservative static subset reported separately",
            "50ms timestamp gate plus ego-pose transform is not a 10cm reference-uncertainty guarantee",
            "Pretraining overlap is unknown; no fully unseen-dataset claim"],
        model_inference_performed=False,scale_fitting_performed=False,training_performed=False,phone_app_changed=False)
    manifest["manifest_sha256"]=common.identity(manifest)
    common.immutable(output/"input_manifest.json",common.encoded(manifest))
    common.immutable(output/"inventory.json",common.encoded(inventory))
    print(json.dumps(inventory,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--output",type=Path,required=True)
    prepare(p.parse_args())


if __name__=="__main__":main()
