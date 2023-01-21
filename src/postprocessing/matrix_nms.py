import torch
from torch import Tensor
from typing import Tuple

def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
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



# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / (union + 1e-16)
    return iou


def matrix_nms(boxes, scores, method = 'gaussian', sigma = 0.5):

	'''https://arxiv.org/pdf/2003.10152.pdf'''
	# boxes (Tensor[N, 4])

	'''boxes must be sorted such that scores are in descending order'''

	# calculate all pair-wise iou's
	ious = box_iou(boxes, boxes)       # N x N 

	# take upper traingular
	ious = torch.triu(ious, diagonal = 1)

	# max IoU for each: NxN
	ious_cmax = ious.max(dim = 0)[0]
	ious_cmax = ious_cmax.expand_as(ious).T
	
	
	if method == 'gaussian': # gaussian
		decay = torch.exp(-1.0 * (ious ** 2 - ious_cmax ** 2) / sigma)
	else: # linear
		decay = (1 - ious) / (1 - ious_cmax)
	
	# decay factor: N
	decay = decay.min(dim = 0)[0]
	
	return scores * decay 


def nms(boxes, scores, idxs, sigma = 0.5, nms_threshold = 0.9):

	'''use the coordinate trick described in:
		https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
	'''

	# boxes (float) (Tensor[N, 4])  # x1 y1 x2 y2
	# scores (float) (Tensor[N])    # [0, 1] => denoting a confidence score derived from 
	                                # class probability and/or localization quality
	# idxs (int; may require to convert to float) (Tensor[N]) # class id of the box

	# nms_threshold (float) (scalar) # [0, 1] => threshold to reject non-maximal boxes; default 0.3

	# boxes must be sorted such that scores are in descending order'''

	# will return indices for boxes to keep (top_k boxes with score above nms_threshold)

	scores = scores.to(boxes)

	if boxes.numel() == 0:
		return torch.empty((0,), dtype=torch.int64, device=boxes.device)


	max_coordinate = boxes.max()

	# strategy: in order to perform NMS independently per class,
    # add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap


	offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
	boxes_for_nms = boxes + offsets[:, None]
	
	'''now call the matrix_nms routine to find the decayed scores'''
	scores_decayed = matrix_nms(boxes_for_nms, scores, method = 'gaussian', sigma = sigma)
	
	'''threshold the decayed scores to find indices of boxes to keep'''
	indices = torch.arange(len(boxes)).to(torch.int64).to(boxes.device)
	keep = indices[(scores_decayed > nms_threshold).nonzero()]       # integer indices
	
	return keep


def keep_top_k(boxes, scores, idxs, top_k = 8):

	boxes_keep = []
	scores_keep = []
	idxs_keep = []

	unique_idxs = torch.unique(idxs)

	for idx in unique_idxs:

		mask = (idxs == idx)
		box = boxes[mask]
		score = scores[mask]
		ids = idxs[mask]

		if len(box) > top_k:
			sort_idx = score.argsort(descending = True)
			score = score[sort_idx]
			box = box[sort_idx]
			ids = ids[sort_idx]

			score = score[:top_k]
			box = box[:top_k]
			ids = ids[:top_k]

		boxes_keep.append(box)
		scores_keep.append(score)
		idxs_keep.append(ids)

	return torch.cat(boxes_keep), torch.cat(scores_keep), torch.cat(idxs_keep)

	'''
	if len(boxes) > top_k:

		#sort again in descending order of score

		sort_idx = scores.argsort(descending = True)
		scores = scores[sort_idx]
		boxes = boxes[sort_idx]
		idxs = idxs[sort_idx]
		
		scores = scores[:top_k]
		boxes = boxes[:top_k]
		idxs = idxs[:top_k]

	return boxes, scores, idxs
	'''

    
    