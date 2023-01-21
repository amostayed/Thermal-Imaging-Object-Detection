#encoding:utf-8
#
#created by xiongzihua 2017.12.26
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.training.box_utils import IoULoss, bbox_iou

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


''' not used '''
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, cls_weights = None, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.cls_weights = cls_weights
        self.epsilon = epsilon
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(weight = self.cls_weights, reduction = self.reduction)

    def forward(self, preds, target):
        ''''input types similar to nn.CrossEntropyLoss()

            preds - logits (Float tensor) size: [Batch x No. of classes]
            target - class index (Long tensor) size; [Batch]

        '''
        n = preds.size()[-1]                                           ## number of categories
        log_preds = F.log_softmax(preds, dim=-1)                       ## log-probability
        
        if self.cls_weights is None:
            loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction) 
        else:
            #self.cls_weights = self.cls_weights.to(log_preds.device)
            if self.reduction == 'mean':
                loss = (-(log_preds * self.cls_weights).sum(dim=-1)).sum() / self.cls_weights[target].sum()   ## weighted mean
            else:
                loss = (-(log_preds * self.cls_weights).sum(dim=-1)).sum()


        nll = self.nll_loss(log_preds, target)                         ## negative log-likelyhood loss (weighted) 
        #
        '''weighted sum of log-probability & NLL loss'''
        return linear_combination(loss / n, nll, self.epsilon)




##########################################################################

class yoloLoss(nn.Module):
    
    def __init__(self, total_epochs, num_classes, image_size, anchors, alpha = 1.0, l_coord = 5, l_obj = 1, l_noobj = 0.5, l_cls = 1., l_loc = 1.):
        super(yoloLoss,self).__init__()
        
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.anchors = anchors
        self.alpha = alpha

        self.l_coord = l_coord
        self.l_noobj = l_noobj

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

        self.box_loss = IoULoss(format = 'xywh', method = 'CIoU', reduction='mean')

        self.logistic_loss = nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        #self.logistic_loss = LabelSmoothingCrossEntropy(cls_weights = None, epsilon = 0.1, reduction='mean')

        self.coord_scale = l_coord
        self.obj_scale = l_obj
        self.noobj_scale = l_noobj

        self.cls_scale = l_cls
        self.loc_scale = l_loc

        self.max_epochs = total_epochs
        
    

    def _compute_loss(self, anchor, prediction, target, obj_mask, no_obj_mask):

        #####################################
        '''process the target tensor first'''

        '''exapnd the dimension of the target'''
        #target = target.unsqueeze(dim = 1).expand(-1, 3, -1, -1, -1)

        #print(target.shape, prediction.shape)


        num_attrs = 5 + self.num_classes

        grid_size = target.size(dim = 2)
        stride = self.image_size // grid_size


        anchor_mask = obj_mask[..., : 2]
        anchor = torch.tensor(anchor) / stride                         # normalized anchors: 5 x 2

        if target.is_cuda:
            anchor = anchor.cuda()

        anchor = anchor.unsqueeze(dim = 1).unsqueeze(dim = 1)          # 5 x 1 x 1 x 2
        anchor = anchor.expand_as(anchor_mask)                         # 16 x 5 x 13 x 13 x 2

        
        ''' retrieve responsible anchors & corresponding grid offsets'''
        anchors = anchor[anchor_mask].view(-1, 2).contiguous()

        '''create an array of grid offset'''
        grid = torch.arange(start = 0, end = grid_size)
        y, x = torch.meshgrid(grid, grid)                                  # different from np.meshgrid()
        x_y_offset = torch.stack((x, y), dim = -1)                         # e.g. 13 x 13 x 2
        x_y_offset = x_y_offset.unsqueeze(dim = 0).unsqueeze(dim = 0)      # e.g. 1 x 1 x 13 x 13 x 2

        if target.is_cuda:
            x_y_offset = x_y_offset.cuda()

        x_y_offset = x_y_offset.expand_as(anchor_mask)                     # e.g. 16 x 5 x 13 x 13 x 2

        x_y_offset = x_y_offset[anchor_mask].view(-1, 2).contiguous()

        
        '''find the coordinates at the locations of responsible anchors'''
        coord_target = target[obj_mask].view(-1, num_attrs).contiguous()[:,:4]
        tx, ty = coord_target[:, 0], coord_target[:, 1]
        '''make a change here'''
        tw, th = coord_target[:, 2], coord_target[:, 3]
        #tw, th = torch.exp(coord_target[:, 2] / 2), torch.exp(coord_target[:, 3] / 2)

        txy = (self.alpha * coord_target[:, :2] - (self.alpha - 1) / 2) + x_y_offset
        twh = torch.exp(coord_target[:, 2:]) * anchors 
        tbox = torch.cat((txy, twh), dim = 1)

        '''find the objectness score at the locations of responsible grids'''
        conf_obj_target = target[obj_mask].view(-1, num_attrs).contiguous()[:, 4]

        '''find the objectness score at the locations of not-responsible grids'''
        conf_no_obj_target = target[no_obj_mask].view(-1, num_attrs).contiguous()[:, 4]

        '''find the class index at the locations of responsible anchors;should be long tensor'''
        if self.num_classes == 1:
            class_target = target[obj_mask].view(-1,  num_attrs).contiguous()[:, 5:]      # probability
        else:
            class_target = torch.argmax(target[obj_mask].view(-1, num_attrs).contiguous()[:, 5:], dim =  1)   # category
        

        #print(tx)

        #####################################

        '''now process the prediction tensor '''
        obj_mask = obj_mask[..., 0].unsqueeze(dim = -1).expand_as(prediction)
        no_obj_mask = no_obj_mask[..., 0].unsqueeze(dim = -1).expand_as(prediction)

        '''find the coordinates at the locations of responsible anchors'''
        coord_prediction = prediction[obj_mask].view(-1, num_attrs + 1).contiguous()[:,:4]
        x, y = torch.sigmoid(coord_prediction[:, 0]), torch.sigmoid(coord_prediction[:, 1])
        '''make a change here'''
        #w, h = torch.exp(coord_prediction[:, 2] / 2), torch.exp(coord_prediction[:, 3] / 2)
        w, h = coord_prediction[:, 2], coord_prediction[:, 3]

        xy = (self.alpha * torch.sigmoid(coord_prediction[:, :2]) - (self.alpha - 1) / 2) + x_y_offset                           
        wh = torch.exp(coord_prediction[:, 2:]) * anchors                      ## width-height
        box = torch.cat((xy, wh), dim = 1)        

        '''find the objectness score at the locations of responsible grids'''
        conf_obj_prediction = torch.sigmoid(prediction[obj_mask].view(-1, num_attrs + 1).contiguous()[:, 4])

        '''find the objectness score at the locations of not-responsible grids'''
        conf_no_obj_prediction = torch.sigmoid(prediction[no_obj_mask].view(-1, num_attrs + 1).contiguous()[:, 4])

        '''find the class scores (real-valued) at the locations of responsible anchors'''
        class_prediction = prediction[obj_mask].view(-1, num_attrs + 1).contiguous()[:, 6:]      # logit

        with torch.no_grad():
            loc_target = bbox_iou(box, tbox, x1y1x2y2 = False).clamp(min = 0, max = 1.0)
        
        loc_prediction = prediction[obj_mask].view(-1, num_attrs + 1).contiguous()[:, 5]

        #####################################


        '''now calculate the losses'''

        #loss_x = self.mse_loss(x, tx)
        #loss_y = self.mse_loss(y, ty)
        #loss_w = self.mse_loss(w, tw)
        #loss_h = self.mse_loss(h, th)
        #loss_coord = loss_x + loss_y + loss_w + loss_h

        loss_coord = self.box_loss(box, tbox)

        loss_conf_obj = self.mse_loss(conf_obj_prediction, conf_obj_target)
        loss_conf_noobj = self.mse_loss(conf_no_obj_prediction, conf_no_obj_target)
        
        loss_cls = self.logistic_loss(class_prediction, class_target)

        loss_loc = self.bce_loss(loc_prediction, loc_target)
    
        #total_loss =  loss_coord + loss_conf + loss_cls



        return loss_coord, loss_conf_obj, loss_conf_noobj, loss_cls, loss_loc



    def forward(self, curr_epoch, prediction, target, obj_mask, no_obj_mask):
        
        ###
        loss_coord = 0.0
        loss_conf_obj = 0.0
        loss_conf_noobj = 0.0
        loss_cls = 0.0
        loss_loc = 0.0
        #total_loss = 0.0

        count = 0

        for _prediction, _target, _obj_mask, _no_obj_mask, _anchor in zip(prediction, target, obj_mask, no_obj_mask, self.anchors):

            if not _obj_mask.type(torch.float32).sum().item() == 0:

                #print(_obj_mask.sum())

                _loss_coord, _loss_conf_obj, _loss_conf_noobj, _loss_cls, _loss_loc = self._compute_loss(_anchor, _prediction, _target, _obj_mask, _no_obj_mask)

                loss_coord += _loss_coord
                loss_conf_obj += _loss_conf_obj
                loss_conf_noobj += _loss_conf_noobj
                loss_cls += _loss_cls
                loss_loc += _loss_loc

                count += 1

        
        if curr_epoch < self.max_epochs // 4:
            total_loss = self.coord_scale * loss_coord + self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj + self.cls_scale * loss_cls

        else:
            total_loss = self.coord_scale * loss_coord + self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj + self.cls_scale * loss_cls + self.loc_scale * loss_loc
        

        #print('......................')
        return total_loss, (loss_coord.item() / count, loss_conf_obj.item() / count, loss_conf_noobj.item() / count, loss_cls.item() / count, loss_loc.item() / count)
