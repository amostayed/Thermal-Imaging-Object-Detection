import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import models

import numpy as np

from src.postprocessing.matrix_nms import nms, keep_top_k


class Prediction(nn.Module):
    
    def __init__(self, anchors, alpha = 1.0, beta = 0.5, inp_dim = 416, num_classes = 20, obj_thres = 0.9, conf_thres = 0.5, loc_thres = 0.8, nms_thres = 0.3, size_thres = 0.00, sigma = 0.5, top_k = 8, CUDA = True):

        super(Prediction, self).__init__()

        self.num_classes = num_classes
        self.inp_dim = inp_dim
        #self.anchors =   [[(81, 82), (135, 169), (344, 319)], [(10, 14), (23, 27), (37, 58)]]
        self.anchors =   anchors
        self.alpha = alpha
        self.beta = beta

        self.obj_thres = obj_thres
        self.conf_thres = conf_thres
        self.loc_thres = loc_thres
        self.nms_thres = nms_thres
        self.top_k = top_k
        self.size_thres = size_thres
        self.sigma = sigma

        self.CUDA = CUDA
        
    def bbox_iou(self, box1, box2, x1y1x2y2=True):

        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0
        )
        
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def xywh2xyxy(self, box_in):

        box_out = box_in.new(box_in.shape)

        box_out[..., 0] = box_in[..., 0] - box_in[..., 2] / 2
        box_out[..., 1] = box_in[..., 1] - box_in[..., 3] / 2
        box_out[..., 2] = box_in[..., 0] + box_in[..., 2] / 2
        box_out[..., 3] = box_in[..., 1] + box_in[..., 3] / 2

        return box_out 

    

    def raw_detection(self, predictions):

        """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
        """

        # From (center x, center y, width, height) to (x1, y1, x2, y2)

        predictions[..., :4] = self.xywh2xyxy(predictions[..., :4]).clamp(min = 0, max = self.inp_dim)
        
        output = [None for _ in range(len(predictions))]

        for image_i, image_pred in enumerate(predictions):
            
            # Filter out confidence scores below threshold
            
            image_pred = image_pred[image_pred[:, 4] >= self.obj_thres]

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue

            image_pred = image_pred[(image_pred[:, 2] - image_pred[:, 0]) * (image_pred[:, 3] - image_pred[:, 1]) >= self.inp_dim * self.inp_dim * self.size_thres]
            
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue

            image_pred = image_pred[(image_pred[:, 4] * image_pred[:, 6:].max(dim = 1)[0]) >= self.conf_thres]
            #image_pred = image_pred[image_pred[:, 5:].max(dim = 1)[0] >= self.conf_thres]

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue


            image_pred = image_pred[image_pred[:, 5] >= self.loc_thres]

            

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue

            class_confs, class_preds = image_pred[:, 6:].max(dim = 1, keepdim = True)
            detections = torch.cat((image_pred[:, :6], class_confs.float(), class_preds.float()), dim = 1)

            output[image_i] = detections

            #print(output)


        return output    




    def non_max_suppression(self, raw_detections):

        """
        raw_predictions is a two-level list
        """

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        
        #print()
        output = [None for _ in range(len(raw_detections[0]))]
        #print(output)

        for img_i, (img_det_lev1, img_det_lev2, img_det_lev3) in enumerate(zip(*raw_detections)):
            
            
            #print(type(img_det_lev1))
            #print('........')
            #print(type(img_det_lev2))


            #print((not isinstance(img_det_lev1, torch.Tensor)) and (not isinstance(img_det_lev2, torch.Tensor)))

            #(not isinstance(img_det_lev1, torch.Tensor)) and (not isinstance(img_det_lev2, torch.Tensor))


            '''skip if no detections'''
            if (not isinstance(img_det_lev1, torch.Tensor)) and (not isinstance(img_det_lev2, torch.Tensor)) and (not isinstance(img_det_lev3, torch.Tensor)):
                #print('here1')
                continue
            ####
            if (not isinstance(img_det_lev1, torch.Tensor)) and (isinstance(img_det_lev2, torch.Tensor)) and (not isinstance(img_det_lev3, torch.Tensor)):
                #print('here2')
                img_det = img_det_lev2

            if (not isinstance(img_det_lev2, torch.Tensor)) and (isinstance(img_det_lev1, torch.Tensor)) and (not isinstance(img_det_lev3, torch.Tensor)):
                #print('here3')
                img_det = img_det_lev1

            if (not isinstance(img_det_lev1, torch.Tensor)) and (isinstance(img_det_lev3, torch.Tensor)) and (not isinstance(img_det_lev2, torch.Tensor)):
                #print('here3')
                img_det = img_det_lev3
            ####
            if (isinstance(img_det_lev1, torch.Tensor)) and (isinstance(img_det_lev2, torch.Tensor)) and (not isinstance(img_det_lev3, torch.Tensor)):
                #print('here4')
                img_det = torch.cat((img_det_lev1, img_det_lev2), dim = 0)

            if (isinstance(img_det_lev1, torch.Tensor)) and (isinstance(img_det_lev3, torch.Tensor)) and (not isinstance(img_det_lev2, torch.Tensor)):
                #print('here4')
                img_det = torch.cat((img_det_lev1, img_det_lev3), dim = 0)

            if (isinstance(img_det_lev2, torch.Tensor)) and (isinstance(img_det_lev3, torch.Tensor)) and (not isinstance(img_det_lev1, torch.Tensor)):
                #print('here4')
                img_det = torch.cat((img_det_lev2, img_det_lev3), dim = 0)

            ####
            if (isinstance(img_det_lev1, torch.Tensor)) and (isinstance(img_det_lev2, torch.Tensor)) and (isinstance(img_det_lev3, torch.Tensor)):
                #print('here4')
                img_det = torch.cat((img_det_lev1, img_det_lev2, img_det_lev3), dim = 0)


            boxes = img_det[:,:4]
            scores = ((img_det[:,4] * img_det[:,6]) ** self.beta) * (img_det[:,5] ** (1 - self.beta))
            #scores = img_det[:,5]

            idxs = img_det[:,7].to(torch.int64)

            '''sort in descending order'''
            sort_idx = scores.argsort(descending = True)
            scores = scores[sort_idx]
            boxes = boxes[sort_idx]
            idxs = idxs[sort_idx]

            keep = nms(boxes, scores, idxs, sigma = self.sigma, nms_threshold = self.nms_thres).squeeze()

            if not keep.numel() == 0:   # keep can't be None though if atleast one object is detected

                scores_to_keep = scores[keep]
                boxes_to_keep = boxes[keep]
                idxs_to_keep = idxs[keep]

                boxes_to_keep, scores_to_keep, idxs_to_keep = keep_top_k(boxes_to_keep, scores_to_keep, idxs_to_keep, top_k = self.top_k)

                '''raise the dimension of score & idx by 1'''
                scores_to_keep, idxs_to_keep = scores_to_keep.unsqueeze(dim = -1), idxs_to_keep.unsqueeze(dim = -1)

                
                '''concat along the last dimension'''

                final = torch.cat((boxes_to_keep, scores_to_keep, idxs_to_keep), dim = -1)

                '''if only one box is detected, make sure a 2-D Tensor is returened '''
                if final.dim() < 2:

                    final = final.unsqueeze(dim = 0)

            else:

                final = None
            
            output[img_i] = final

        return output

    
    def _prediction(self, anchors, predictions):

        '''predictions is a tensor'''

        batch_size = predictions.size(dim = 0)
        stride =  self.inp_dim // predictions.size(dim = 2)
        grid_size = self.inp_dim // stride
        bbox_attrs = 6 + self.num_classes
        num_anchors = len(anchors)
    
        predictions = predictions.reshape(batch_size, -1, bbox_attrs)
        
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

        #Sigmoid the  centre_X, centre_Y. and object confidencce
        predictions[:,:,0] = torch.sigmoid(predictions[:,:,0])
        predictions[:,:,1] = torch.sigmoid(predictions[:,:,1])
        predictions[:,:,4] = torch.sigmoid(predictions[:,:,4])
        predictions[:,:,5] = torch.sigmoid(predictions[:,:,5])
    
        #Add the center offsets
        grid = np.arange(grid_size)
        x,y = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(x).view(-1,1)
        y_offset = torch.FloatTensor(y).view(-1,1)

        if self.CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), dim = 1).repeat(num_anchors, 1).unsqueeze(dim = 0)
        #predictions[:,:,:2] += x_y_offset
        predictions[:,:,:2] = self.alpha * predictions[:,:,:2] - 0.5 * (self.alpha - 1) + x_y_offset

        #log space transform height and the width
        anchors = torch.FloatTensor(anchors)
        if self.CUDA:
            anchors = anchors.cuda()

        anchors = anchors.repeat(1, grid_size * grid_size).view(-1, 2).unsqueeze(dim = 0)
    
        predictions[:,:,2:4] = torch.exp(predictions[:,:,2:4]) * anchors
    
        predictions[:,:, 6 : 6 + self.num_classes] = F.softmax((predictions[:,:, 6 : 6 + self.num_classes]), dim = -1)
        #predictions[:,:, 5 : 5 + self.num_classes] = torch.sigmoid((predictions[:,:, 5 : 5 + self.num_classes]))

        predictions[:, :, : 4] *= stride

        # filter out via NMS

        #outputs = self.non_max_suppression(predictions)


        return predictions


    def forward(self, predictions):

        '''predictions is a list of tensors'''

        
        '''make a list of tensors for each level of outputs,
           concatenate them and,
           run NMS

        '''
        outputs = []

        for anchor, prediction in zip(self.anchors, predictions):

            outputs.append(self.raw_detection(self._prediction(anchor, prediction)))


        #print(outputs)
        

        outputs = self.non_max_suppression(outputs)

    
        return outputs