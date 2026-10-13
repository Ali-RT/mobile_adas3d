"""Device-safe, single-group instance GT and selective object-aligned A2 KD.

The pilot optimizes the deployed 50-query path (batch1, frozen BN/dropout).
Instance GT follows MonoDETR's focal/L1/GIoU/Laplace/dimension/angle recipe;
the optional dense depth-map auxiliary loss and training query groups are
not used. No term/module is added to the exported student.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

COMPONENTS = ('classification', 'boxes', 'dimensions', 'depth', 'angle')
GT_WEIGHTS = dict(loss_ce=2., loss_center=10., loss_bbox=5., loss_giou=2.,
                  loss_depth=1., loss_dim=1., loss_angle=1.)
COSTS = dict(classification=2., center=10., box=5., giou=2.)
TEMPERATURE = 2.


def corners(boxes):
    xc, yc, left, right, top, bottom = boxes.unbind(-1)
    return torch.stack((xc-left, yc-top, xc+right, yc+bottom), -1)


def overlaps(a, b, generalized=False):
    aa, bb = corners(a), corners(b)
    area_a = (aa[:, 2:]-aa[:, :2]).clamp_min(0).prod(-1)
    area_b = (bb[:, 2:]-bb[:, :2]).clamp_min(0).prod(-1)
    intersect = (torch.minimum(aa[:, None, 2:], bb[None, :, 2:])-
                 torch.maximum(aa[:, None, :2], bb[None, :, :2])).clamp_min(0).prod(-1)
    union = area_a[:, None]+area_b[None]-intersect
    result = intersect/union.clamp_min(1e-8)
    if generalized:
        enclosing = (torch.maximum(aa[:, None, 2:], bb[None, :, 2:])-
                     torch.minimum(aa[:, None, :2], bb[None, :, :2])).clamp_min(0).prod(-1)
        result = result-(enclosing-union)/enclosing.clamp_min(1e-8)
    return result


def validate(outputs, target):
    shapes = dict(pred_logits=(1,50,3), pred_boxes=(1,50,6), pred_3d_dim=(1,50,3),
                  pred_depth=(1,50,2), pred_angle=(1,50,24))
    for name, shape in shapes.items():
        if outputs[name].shape != shape or outputs[name].dtype!=torch.float32 or not torch.isfinite(outputs[name]).all():
            raise ValueError(f'Invalid deployed FP32 training output: {name}')
    n = len(target['labels'])
    expected = dict(labels=(n,), boxes_3d=(n,6), size_3d=(n,3), depth=(n,1),
                    heading_bin=(n,1), heading_res=(n,1))
    for name, shape in expected.items():
        if target[name].shape != shape or not torch.isfinite(target[name]).all():
            raise ValueError(f'Invalid native GT: {name}')
    if n > 50 or not ((target['labels']==0)|(target['labels']==1)).all():
        raise ValueError('Expected at most50 product Pedestrian/Vehicle targets')
    if n and ((target['size_3d']<=0).any() or (target['depth']<=0).any()
              or (target['boxes_3d'][:, 2:]<0).any()):
        raise ValueError('Invalid native metric/box GT')


@torch.no_grad()
def assignment(outputs, target):
    validate(outputs, target)
    device = outputs['pred_logits'].device
    if not len(target['labels']):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    prob = outputs['pred_logits'][0].sigmoid()
    labels = target['labels'].long()
    negative = .75*prob.square()*(-(1-prob+1e-8).log())
    positive = .25*(1-prob).square()*(-(prob+1e-8).log())
    box = outputs['pred_boxes'][0]; gt = target['boxes_3d']
    cost = (COSTS['classification']*(positive[:, labels]-negative[:, labels])+
            COSTS['center']*torch.cdist(box[:, :2], gt[:, :2], p=1)+
            COSTS['box']*torch.cdist(box[:, 2:], gt[:, 2:], p=1)-
            COSTS['giou']*overlaps(box, gt, True))
    i, j = linear_sum_assignment(cost.cpu().numpy())
    return torch.tensor(i, device=device), torch.tensor(j, device=device)


def alpha(angle):
    bins = angle[:, :12].argmax(-1)
    residual = angle[:, 12:].gather(1, bins[:, None]).squeeze(1)
    return bins*(2*math.pi/12)+residual


def component_errors(outputs, target, queries, objects):
    labels = target['labels'][objects].long()
    desired = F.one_hot(labels, 3).to(outputs['pred_logits'].dtype)
    gt_angle = target['heading_bin'][objects, 0]*(2*math.pi/12)+target['heading_res'][objects, 0]
    difference = alpha(outputs['pred_angle'][0, queries])-gt_angle
    return dict(
        classification=(outputs['pred_logits'][0, queries].sigmoid()-desired).abs().mean(-1),
        boxes=(outputs['pred_boxes'][0, queries]-target['boxes_3d'][objects]).abs().mean(-1),
        dimensions=((outputs['pred_3d_dim'][0, queries]-target['size_3d'][objects])/
                    target['size_3d'][objects].clamp_min(.1)).abs().mean(-1),
        depth=(outputs['pred_depth'][0, queries, 0]-target['depth'][objects, 0]).abs()/target['depth'][objects, 0].clamp_min(.1),
        angle=torch.atan2(difference.sin(), difference.cos()).abs()/math.pi)


def gt_loss(outputs, target):
    i, j = assignment(outputs, target); n = max(len(j), 1)
    logits = outputs['pred_logits'][0]
    labels = target['labels'][j].long()
    desired = torch.zeros_like(logits)
    desired[i, labels] = 1.
    prob = logits.sigmoid(); pt = prob*desired+(1-prob)*(1-desired)
    focal = F.binary_cross_entropy_with_logits(logits, desired, reduction='none')*(1-pt).square()
    parts = dict(loss_ce=(focal*(.25*desired+.75*(1-desired))).sum()/n)
    zero = sum(v.sum()*0 for k,v in outputs.items() if k.startswith('pred_') and torch.is_tensor(v))
    if len(j):
        boxes = outputs['pred_boxes'][0, i]; gt = target['boxes_3d'][j]
        parts['loss_center'] = (boxes[:, :2]-gt[:, :2]).abs().sum()/n
        parts['loss_bbox'] = (boxes[:, 2:]-gt[:, 2:]).abs().sum()/n
        parts['loss_giou'] = (1-overlaps(boxes, gt, True).diag()).sum()/n
        depth = outputs['pred_depth'][0, i]; actual = target['depth'][j, 0]
        parts['loss_depth'] = (1.4142*torch.exp(-depth[:, 1])*(depth[:, 0]-actual).abs()+depth[:, 1]).sum()/n
        dimensions = outputs['pred_3d_dim'][0, i]; actual_dim = target['size_3d'][j]
        relative = (dimensions-actual_dim).abs()/actual_dim
        compensation = ((dimensions-actual_dim).abs().mean()/relative.mean().clamp_min(1e-8)).detach()
        parts['loss_dim'] = (relative*compensation).sum()/n
        angle = outputs['pred_angle'][0, i]; bins = target['heading_bin'][j, 0].long()
        residual = angle[:, 12:].gather(1, bins[:, None]).squeeze(1)
        parts['loss_angle'] = (F.cross_entropy(angle[:, :12], bins, reduction='none')+
                               (residual-target['heading_res'][j, 0]).abs()).sum()/n
    else:
        parts.update({key:zero for key in GT_WEIGHTS if key!='loss_ce'})
    total = sum(parts[key]*weight for key, weight in GT_WEIGHTS.items())
    errors = component_errors(outputs, target, i, j)
    metrics = {name:float(value.detach().mean()) if len(value) else None for name,value in errors.items()}
    return total, parts, metrics


@torch.no_grad()
def teacher_matches(teacher, target, threshold=.1, min_iou=.5):
    i, j = assignment(teacher, target)
    mapping = torch.full((len(target['labels']),), -1, dtype=torch.long, device=teacher['pred_logits'].device)
    if len(j):
        prob = teacher['pred_logits'][0, i].sigmoid()
        labels = target['labels'][j].long()
        keep = ((prob.argmax(-1)==labels)&(prob.gather(1, labels[:, None]).squeeze(1)>=threshold)&
                (overlaps(teacher['pred_boxes'][0, i], target['boxes_3d'][j]).diag()>=min_iou)&
                (teacher['pred_depth'][0, i, 0]>.1)&(teacher['pred_3d_dim'][0, i]>0).all(-1))
        mapping[j[keep]] = i[keep]
    return mapping


def class_mean(values, labels, keep, zero):
    means = [values[keep&(labels==cls)].mean() for cls in (0,1) if (keep&(labels==cls)).any()]
    return torch.stack(means).mean() if means else zero


def kd_loss(student, teacher, target, mapping):
    """Teacher/student query numbers are never assumed to correspond.

    Hard GT remains primary. Each teacher component must be at least10%
    closer to GT than the current student on the same matched object.
    Means are balanced across classes present in that component's mask.
    """
    i, j = assignment(student, target)
    zero = student['pred_logits'].sum()*0
    selected = mapping[j]>=0; i,j = i[selected],j[selected]
    ti = mapping[j]
    if len(j):
        with torch.no_grad():
            se = component_errors(student,target,i,j); te = component_errors(teacher,target,ti,j)
            masks = {name:te[name]+1e-6<=.9*se[name] for name in COMPONENTS}
        labels = target['labels'][j].long(); t = TEMPERATURE
        sl, tl = student['pred_logits'][0,i],teacher['pred_logits'][0,ti].detach()
        p = (tl/t).sigmoid().clamp(1e-6,1-1e-6)
        classification = (F.binary_cross_entropy_with_logits(sl/t,p,reduction='none')+
                           p*p.log()+(1-p)*(1-p).log()).mean(-1).clamp_min(0)*t*t
        boxes = F.smooth_l1_loss(student['pred_boxes'][0,i],teacher['pred_boxes'][0,ti].detach(),reduction='none').mean(-1)
        scale = target['size_3d'][j].clamp_min(.1)
        dimensions = F.smooth_l1_loss(student['pred_3d_dim'][0,i]/scale,teacher['pred_3d_dim'][0,ti].detach()/scale,reduction='none').mean(-1)
        depth = F.smooth_l1_loss(student['pred_depth'][0,i,0].clamp_min(.1).log(),
                                teacher['pred_depth'][0,ti,0].detach().log(),reduction='none')
        sa,ta = student['pred_angle'][0,i],teacher['pred_angle'][0,ti].detach()
        pa = (ta[:,:12]/t).softmax(-1); bins = ta[:,:12].argmax(-1)
        angle = F.kl_div((sa[:,:12]/t).log_softmax(-1),pa,reduction='none').sum(-1).clamp_min(0)*t*t
        angle += F.smooth_l1_loss(sa[:,12:].gather(1,bins[:,None]).squeeze(1),
                                 ta[:,12:].gather(1,bins[:,None]).squeeze(1),reduction='none')
        values = dict(classification=classification,boxes=boxes,dimensions=dimensions,depth=depth,angle=angle)
        parts = {name:class_mean(value,labels,masks[name],zero) for name,value in values.items()}
        coverage = {name:{cls:int((masks[name]&(labels==label)).sum()) for cls,label in [('Pedestrian',0),('Vehicle',1)]} for name in COMPONENTS}
    else:
        parts = {name:zero for name in COMPONENTS}
        coverage = {name:dict(Pedestrian=0,Vehicle=0) for name in COMPONENTS}
    return sum(parts.values()), parts, coverage
