import torch
import numpy as np
import math
import torch.nn as nn


''' implements gIoU, dIoU, and cIoU  
    
    some code snippets are borrowed from torchvision

    cIoU from https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47

'''



class IoULoss(nn.Module):
    
    def __init__(self, format = 'xyxy', method = 'GIoU', reduction='mean'):
        super().__init__()

        self.method = method
        self.format = format
        self.reduction = reduction

    def forward(self, preds, target, weights = None):

        x1y1x2y2 = True if self.format == 'xyxy' else False

        if self.method == 'GIoU':
            GIoU, DIoU, CIoU = True, False, False

        if self.method == 'DIoU':
            GIoU, DIoU, CIoU = False, True, False

        if self.method == 'CIoU':
            GIoU, DIoU, CIoU = False, False, True


        iou = bbox_iou(preds, target, x1y1x2y2 = x1y1x2y2, GIoU = GIoU, DIoU = DIoU, CIoU = CIoU)
        iou = 1.0 - iou   ## loss to minimize

        if self.reduction == 'mean':
            if weights is None:
                iou = iou.mean(dim = -1)
                
            else:
                iou = (iou * weights).sum(dim = -1) / weights.sum(dim = -1)

        if self.reduction == 'sum':
            iou = iou.sum()


        return iou



def _upcast(t: torch.Tensor) -> torch.Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def xywh2xyxy(boxes):

    '''(cx, cy, w, h) to (x1, y1, x2, y2)'''

    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes

    #boxes2 = boxes.detach().clone()

    #boxes2[:, 0] = boxes[:, 0] - boxes[:, 2] / 2, 
    #boxes2[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    #boxes2[:, 1] = boxes[:, 1] - boxes[:, 3] / 2, 
    #boxes2[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    #return boxes2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    
    '''box: N x 4'''

    if not x1y1x2y2:
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    # clip possible negative values
    box1 = box1.clamp(0)
    box2 = box2.clamp(0)

    # find area (N) of two sets of boxes; caution against possible zero area
    area1 = box_area(box1) + eps   # [N,]
    area2 = box_area(box2) + eps   # [N,]

    #print(area1, area2)

    # find intersection of two sets of boxes
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N, 2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]         # [N,]

    # find union (can never be zero)
    union = area1 + area2 - inter       # [N,]

    # basic iou
    iou = inter / union

         
    if GIoU or DIoU or CIoU:
        # the convex hull of two set of boxes
        lt = torch.min(box1[:, :2], box2[:, :2])  # [N,2]
        rb = torch.max(box1[:, 2:], box2[:, 2:])  # [N, 2]

        wh = _upcast(rb - lt).clamp(min=0)  # [N, 2]
        
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1

            c2 = (wh ** 2).sum(dim = 1) + eps  # convex diagonal squared
            rho2 = ((box1[..., 0::2].sum(dim = 1) - box2[..., 0::2].sum(dim = 1)) ** 2 + 
                    (box1[..., 1::2].sum(dim = 1) - box2[..., 1::2].sum(dim = 1)) ** 2) / 4.0 # center distance squared
            
            if DIoU:
                return iou - rho2 / c2  # DIoU
            
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                w2 = box2[..., 2] - box2[..., 0] + eps
                h2 = box2[..., 3] - box2[..., 1] + eps
                w1 = box1[..., 2] - box1[..., 0] + eps
                h1 = box1[..., 3] - box1[..., 1] + eps

                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = wh[:, 0] * wh[:, 1] + eps  # convex area, again caution for possible zero
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou

    return iou

#box1 = torch.tensor([[3, 10., 9, 23], [3, 10., 9, 23]])
#box2 = torch.tensor([[25, 39, 28, 51], [7, 18, 11, 30]])
#bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9)