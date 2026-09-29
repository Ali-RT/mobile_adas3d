"""M63c inference-only replacement of BatchNorm running statistics."""
import torch
from m61_common import array_hash

def state_fingerprints(model):
    return {k: array_hash({k:v}) for k,v in model.state_dict().items()}

def restore_running_statistics(model, source):
    """Change only running_mean/var in BatchNorm modules; never affine weights/counters."""
    replacements = {}
    state = model.state_dict()
    for name,module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        if not module.track_running_stats:
            continue
        for suffix in ("running_mean","running_var"):
            key = (name + "." if name else "") + suffix
            if key not in source:
                raise ValueError(f"Missing source BatchNorm buffer: {key}")
            value = source[key]
            if value.shape != state[key].shape or value.dtype != state[key].dtype:
                raise ValueError(f"BatchNorm shape/dtype mismatch: {key}")
            if not torch.isfinite(value).all() or (suffix == "running_var" and (value < 0).any()):
                raise ValueError(f"Invalid source BatchNorm statistics: {key}")
            replacements[key] = value.detach().clone()
    if not replacements:
        raise ValueError("No tracked BatchNorm statistics found")
    before = state_fingerprints(model)
    with torch.no_grad():
        for key,value in replacements.items():
            state[key].copy_(value.to(state[key].device))
    after = state_fingerprints(model)
    changed = [k for k in before if before[k] != after[k]]
    if not set(changed) <= set(replacements):
        raise RuntimeError("Intervention changed non-statistic state")
    for key,value in replacements.items():
        if not torch.equal(model.state_dict()[key].detach().cpu(), value.cpu()):
            raise RuntimeError(f"Source-statistics copy failed: {key}")
    return dict(replaced_buffer_names=sorted(replacements), changed_buffer_names=sorted(changed),
                replaced_buffers=len(replacements), normalization_modules=len(replacements)//2,
                non_statistic_state_unchanged=True, before_sha256=before, after_sha256=after)
