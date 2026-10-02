"""Exercise real dataset CUDA imports and kernels without optimizer updates."""
import argparse
import json
from pathlib import Path
import sys
from setup_m63_isolated_runtime import inventory,validate_inventory

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    receipt=json.loads(a.receipt.read_text())
    current=inventory(sys.executable);validate_inventory(current)
    if current!=receipt["inventory"]: raise RuntimeError("Runtime inventory changed")
    from m63_common import load_manifest
    from m61_common import sha256,write_json
    m=load_manifest(a.manifest)
    repo=Path(m["student"]["repo"])
    sys.path[:0]=[str(repo),str(repo/"lib/models/monodetr/ops")]
    import torch
    import numpy as np
    import MultiScaleDeformableAttention
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from lib.datasets.kitti.kitti_eval_python.rotate_iou import rotate_iou_gpu_eval
    box=np.array([[0.,0.,2.,4.,0.]],dtype=np.float32)
    iou=rotate_iou_gpu_eval(box,box)
    if not np.allclose(iou,1.,atol=1e-5): raise RuntimeError("CUDA IoU smoke failed")
    x=torch.ones(4,device="cuda",requires_grad=True)
    (x*x).sum().backward();torch.cuda.synchronize()
    if not torch.equal(x.grad,torch.full_like(x,2)): raise RuntimeError("CUDA backward failed")
    from numba import cuda
    report=dict(complete=True,optimizer_steps=0,dataset_import=True,iou=iou.tolist(),
        torch_cuda_backward=True,numba_cuda_module=cuda.__file__,
        runtime_receipt_sha256=sha256(a.receipt),manifest_sha256=m["manifest_sha256"],
        historical_runtime_equivalence_claimed=False)
    write_json(a.output,report)
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
