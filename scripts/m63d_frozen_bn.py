"""M63d: freeze only BatchNorm mode; retain original parameter trainability."""
import torch
from m61_common import array_hash

def freeze_bn_statistics(model):
    modules=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            if not module.track_running_stats:
                raise ValueError(f"Untracked BatchNorm cannot use source statistics: {name}")
            module.eval()
            modules.append(name)
    if not modules: raise ValueError("No BatchNorm modules found")
    return modules

def bn_fingerprints(model):
    state={}
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            for suffix in ("running_mean","running_var","num_batches_tracked"):
                value=getattr(module,suffix)
                if value is None: raise ValueError("Missing tracked BatchNorm state")
                key=(name+"." if name else "")+suffix
                state[key]=array_hash({key:value})
    return state

def assert_bn_frozen(model, expected=None):
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) and module.training:
            raise RuntimeError("BatchNorm returned to training mode")
    if expected is not None and bn_fingerprints(model)!=expected:
        raise RuntimeError("Frozen BatchNorm statistics/counters changed")
