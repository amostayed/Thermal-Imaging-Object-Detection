'''
modified from original code by Zihua
https://github.com/xiongzihua/pytorch-YOLO-v1/blob/master/dataset.py
Major modifications are:
1. Keep aspect ratio while resize/scaling
2. Image normalization changed to obtain zero mean, unit variance (originally mean subtraction)
3. bounding box clamping (may be not important)
'''
# TODO : clean up the code
import os
import sys
import os.path
#
import random
import numpy as np
#
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#
import cv2
#
class ThermalDataset(data.Dataset):
    #
    
    #######
    def __init__(self, root, list_file, image_size, anchors, num_classes = 1, alpha = 1.0, train = True, transform = None):
        #
        print('data init')
        self.image_size = image_size       # training image size
        self.mosaic_border = [-image_size // 2, -image_size // 2]
        self.anchors = anchors             # anchor boxes
        self.size_range = [128, 64, 0]
        self.anchor_t = 4.
        self.num_classes = num_classes
        self.stride = 32
        self.root = root                   # root directory of the jpg images
        self.train = train                 # bool to indicate train or validation
        self.transform = transform
        self.fnames = []                   # all the training file names accumulated here
        self.boxes = []
        self.labels = []
        self.epsilon = 1e-16
        self.mean = (135, 135, 135)        # data set mean for RGB channels
                                           # these values will be used for padding
                                           # and eventually after normalization the padding 
                                           # values will be zero
        
        self.alpha = alpha
        #self.size_weights = [3.0, 2.0, 1.0]


        #
        '''read all the lines'''
        with open(list_file) as f:         # list_file is the annotation text file for the images in root directory
            lines  = f.readlines()
        #
        for line in lines:
            if line.startswith('#'):
                continue
            splited = line.strip().split()
            if not splited or len(splited) < 8:
                #print(splited[0], "no box")
                continue
            #
            name = splited[0]
            
            #
            #splited = line.strip().split()
            #self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 7
            box = []
            label = []
            for i in range(num_boxes):
                '''coordinates'''
                x = float(splited[3 + 7 * i])
                y = float(splited[4 + 7 * i])
                x2 = float(splited[5 +  7* i])
                y2 = float(splited[6 + 7 * i])
                '''class'''
                c = splited[7 + 7 * i]

                if int(c) > - 1:
                    box.append([x, y, x2, y2])
                    label.append(int(c) + 1)

            if len(box) > 0:
                self.boxes.append(torch.Tensor(box))           # float 32
                self.labels.append(torch.LongTensor(label))    # int 64
                self.fnames.append(name)

            
        #
        self.num_samples = len(self.boxes)                 # total number of annotated bounding boxes
        self.indices = range(len(self.boxes))              # indices of the imagers which will be later used in getitem() 

                                                           # and mosaicing
    #########
    def __getitem__(self, idx):
        #
        '''this is the method for retrieving data with the iterator'''
        #
        index = self.indices[idx]
        #print(self.fnames[index])
        #print(index)
        
        if self.train:                                     # data augmentation only during training
            ''' all these are done in BGR'''
            
            '''image-depended mosaicing'''

            #boxes = self.boxes[index].clone()
            #mosaic_prob = 1.0 - self.image_weight_from_size(boxes)
            mosaic_prob = 0.5

            mosaic = random.random() < mosaic_prob

            if mosaic: 
                img, boxes, labels = self.load_mosaic_4(index)  # composite image

                
                if random.random() < 0.1:
                    img2, boxes2, labels2 = self.load_mosaic_4(random.randint(0, self.num_samples - 1))
                    r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                    img = (img * r + img2 * (1 - r)).astype(np.uint8)
                    labels = torch.cat((labels, labels2), dim = 0)
                    boxes = torch.cat((boxes, boxes2), dim = 0)

                

            else:
                img, _, boxes, labels = self.load_image(index)  # single image 

                
                if random.random() < 0.1:
                    img2, boxes2, labels2 = self.load_mosaic_4(random.randint(0, self.num_samples - 1))
                    r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                    img = (img * r + img2 * (1 - r)).astype(np.uint8)
                    labels = torch.cat((labels, labels2), dim = 0)
                    boxes = torch.cat((boxes, boxes2), dim = 0)
            
            
            img, boxes, labels = self.randomShift(img,boxes,labels)
            img, boxes, labels = self.randomZoom(img, boxes, labels, is_mosaic = mosaic)
            
            img, boxes = self.random_flip(img, boxes)
            img = self.randomBlur(img)

            img = self.hsv_augment(img)

            if random.random() < 0.5:
                img = self.hist_equalize(img)

        #

        else:   # test/val

            img, _, boxes, labels = self.load_image(index)  # single image with no augmentation
            
        '''resize to network dimension
        '''
        img, boxes = self.Resize(img, boxes)

        '''
        convert BGR to RGB, since pret-trained models use RGB
        '''
        
        img = self.BGR2RGB(img)
        
        '''
        fix b-box size if required
        '''
        boxes = boxes.clamp(0, self.image_size)            # keeping the box within the bound of the image
        
        '''important  !!!!!!!
           unusual boxes, such as, less than 2 pixels in either dimension,
           or unusually narrow or long, are excluded
        '''
        invalid = self.invalid_mask(boxes)
        boxes = boxes[invalid.expand_as(boxes)].view(-1, 4)
        labels = labels[invalid.view(-1)]

        
        '''
        weights = torch.zeros((boxes.shape[0], 1))
        for i, box in enumerate(boxes):

            area = (box[2] - box[0]) * (box[3] - box[1])
            
            if  area <= 32 ** 2:
                weights[i] = 3.0

            if  area > 32 ** 2 and area <= 96 ** 2:
                weights[i] = 2.0

            if  area > 96 ** 2:
                weights[i] = 1.0

        boxes = torch.cat([boxes, weights], dim = 1)
        '''

        ##### encoding


        #target, mask_obj, box_weight  = self.encoder(boxes, labels)
        target, mask_obj, mask_no_obj = self.encoder(boxes, labels)

        #print(boxes.shape)
        #print(labels.unsqueeze(dim = 1).shape)

        boxes = torch.cat((boxes, labels.float().unsqueeze(dim = 1) - 1.0), dim = 1)
        gt_boxes = torch.zeros((len(boxes), 6))
        gt_boxes[:, 1:] = boxes
        #
        if self.transform:
            img_n = self.transform(img)

        else:
            img_n = torch.tensor(img.copy())
        #
        #return img, boxes, labels, img_n, target
        return gt_boxes, img_n, target, mask_obj, mask_no_obj

    def collate_fn(self, batch):

        boxes, imgs, targets, mask_obj, mask_no_obj = list(zip(*batch))
        
        # Remove empty placeholder targets
        gt_boxes = [box for box in boxes if box is not None]
        # Add sample index to targets
        for i, box in enumerate(gt_boxes):
            box[:, 0] = i
        gt_boxes = torch.cat(gt_boxes, dim = 0)


        imgs = torch.stack(imgs)

        '''targets: a list of lists
           number of lists inside equals batch_size,
           each list has exactly two tensors (13 x 13 & 26 x 26)

        '''
        targets = list(zip(*targets))
        mask_obj = list(zip(*mask_obj))
        mask_no_obj = list(zip(*mask_no_obj))

        targets = [torch.stack(item) for item in targets]
        mask_obj = [torch.stack(item) for item in mask_obj]
        mask_no_obj = [torch.stack(item) for item in mask_no_obj]

        
        #return gt_boxes, imgs, targets, mask_obj, mask_no_obj
        return imgs, gt_boxes, targets, mask_obj, mask_no_obj

    ###########
    def __len__(self):
        return self.num_samples
    
    #
    def load_image(self, idx):

        fname = self.fnames[idx]

        #print(fname)
        '''read image (BGR) and b-boxes'''
        img = cv2.imread(os.path.join(self.root, fname))
        #print(img.shape)
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        #print(boxes.shape)

        if len(boxes) == 0:
            print('problem')

        #print('%%%%%%%%%%%%%%%%%%%%%%%%%')
        #
        return img, img.shape[:2], boxes, labels
    #

    def image_weight_from_size(self, boxes):
    
        weights = torch.zeros((boxes.shape[0], 1))
        for i, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
                
            if  area <= 32 ** 2:
                weights[i] = 0.5

            if  area > 32 ** 2 and area <= 96 ** 2:
                weights[i] = 1.0 / 3.0

            if  area > 96 ** 2:
                weights[i] = 1.0 / 6.0

        return weights.mean().item()

    #

    def load_mosaic(self, idx):

        color = self.mean
        color = color[::-1] 
        
        boxes4, labels4 = [], []
        s = self.image_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [idx] + [self.indices[random.randint(0, self.num_samples - 1)] for _ in range(3)]  # 3 additional image indices

        
        for i, index in enumerate(indices):
            # Load image
            img, _, boxes, labels  = self.load_image(index)
            #img, boxes = self.resizeMosaic(img, boxes)
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), color, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            #
            #mask = self.get_mask(boxes, x1b, y1b, x2b, y2b)
            #boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
            #labels = labels[mask.view(-1)]
            #
            if boxes.size:  ## if all boxes are not expelled
                boxes = self.xywhn2xyxy(boxes, padw, padh) 

                boxes4.append(boxes)
                labels4.append(labels)

        boxes4 = torch.cat(boxes4, 0)
        labels4 = torch.cat(labels4, 0)

        #boxes4 = boxes4.clamp(0, 2 * s)

        
        #
        return img4, boxes4, labels4

    def load_mosaic_2(self, idx):

        color = self.mean
        color = color[::-1] 
        
        boxes2, labels2 = [], []
        s = self.image_size
        yc = 0
        xc = int(random.uniform(0, 640))  # mosaic center x, y
        #print(xc)
        indices = [idx] + [self.indices[random.randint(0, self.num_samples - 1)] for _ in range(1)]  # 3 additional image indices

        
        for i, index in enumerate(indices):
            # Load image
            img, _, boxes, labels  = self.load_image(index)
            #print(boxes.shape)
            #img, boxes = self.resizeMosaic(img, boxes)
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img2 = np.full((h, w * 2, img.shape[2]), color, dtype=np.uint8)  # base image with 2 tiles

                img2[:, 0 : w] = img

                plt.imshow(img2)
                plt.show()

                #print(boxes)

                boxes[:, 0] = torch.min(torch.max(torch.tensor(0), boxes[:, 0] + 0 - xc), torch.tensor(w - 1)) # top left x
                boxes[:, 1] = torch.min(torch.max(torch.tensor(0), boxes[:, 1] + 0 - yc), torch.tensor(h - 1))  # top left y
                boxes[:, 2] = torch.min(torch.max(torch.tensor(0), boxes[:, 2] + 0 - xc), torch.tensor(w - 1)) # bottom right x
                boxes[:, 3] = torch.min(torch.max(torch.tensor(0), boxes[:, 3] + 0 - yc), torch.tensor(h - 1))  # bottom right y

                mask = self.get_mask(boxes, 0, 0, w, h)
                boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
                labels = labels[mask.view(-1)]

            elif i == 1:  # top right
                img2[:, w : ] = img

                plt.imshow(img2)
                plt.show()

                boxes[:, 0] = torch.min(torch.max(torch.tensor(0), boxes[:, 0] + w - xc), torch.tensor(w - 1)) # top left x
                boxes[:, 1] = torch.min(torch.max(torch.tensor(0), boxes[:, 1] + 0 - yc), torch.tensor(h - 1))  # top left y
                boxes[:, 2] = torch.min(torch.max(torch.tensor(0), boxes[:, 2] + w - xc), torch.tensor(w - 1))  # bottom right x
                boxes[:, 3] = torch.min(torch.max(torch.tensor(0), boxes[:, 3] + 0 - yc), torch.tensor(h - 1))  # bottom right y


                print(boxes)

                mask = self.get_mask(boxes, 0, 0, w, h)
                boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
                labels = labels[mask.view(-1)]

                print(boxes)

            #print(img2.shape)
            #print(boxes)

            
            #print(boxes)
            
            if boxes.size:  ## if all boxes are not expelled
                #boxes = self.xywhn2xyxy(boxes, padw, padh) 

                
                boxes2.append(boxes)
                labels2.append(labels)

        img2 = img2[:, xc:xc+w]
        plt.imshow(img2)
        plt.show()

        boxes2 = torch.cat(boxes2, 0)
        labels2 = torch.cat(labels2, 0)

        #boxes4 = boxes4.clamp(0, 2 * s)

        
        #
        return img2, boxes2, labels2

    def load_mosaic_4(self, idx):

        color = self.mean
        color = color[::-1] 
        
        boxes4, labels4 = [], []
        
        yc = int(random.uniform(0, 512))
        xc = int(random.uniform(0, 640))  # mosaic center x, y
        #print(xc)
        indices = [idx] + [self.indices[random.randint(0, self.num_samples - 1)] for _ in range(3)]  # 3 additional image indices

        for i, index in enumerate(indices):
            # Load image
            img, _, boxes, labels  = self.load_image(index)
            #img, boxes = self.resizeMosaic(img, boxes)
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((h * 2, w * 2, img.shape[2]), color, dtype=np.uint8)  # base image with 2 tiles

                img4[: h, : w] = img

                boxes[:, 0] = boxes[:, 0] + 0 - xc
                boxes[:, 1] = boxes[:, 1] + 0 - yc 
                boxes[:, 2] = boxes[:, 2] + 0 - xc
                boxes[:, 3] = boxes[:, 3] + 0 - yc


            elif i == 1:  # top right
                img4[: h, w : ] = img

                boxes[:, 0] = boxes[:, 0] + w - xc
                boxes[:, 1] = boxes[:, 1] + 0 - yc
                boxes[:, 2] = boxes[:, 2] + w - xc
                boxes[:, 3] = boxes[:, 3] + 0 - yc


            elif i == 2:  # bottom left
                img4[h :, : w] = img

                boxes[:, 0] = boxes[:, 0] + 0 - xc
                boxes[:, 1] = boxes[:, 1] + h - yc
                boxes[:, 2] = boxes[:, 2] + 0 - xc
                boxes[:, 3] = boxes[:, 3] + h - yc


            elif i == 3:  # bottom right
                img4[h :, w : ] = img

                boxes[:, 0] = boxes[:, 0] + w - xc
                boxes[:, 1] = boxes[:, 1] + h - yc
                boxes[:, 2] = boxes[:, 2] + w - xc
                boxes[:, 3] = boxes[:, 3] + h - yc

            
            boxes[:, 0] = torch.min(torch.max(torch.tensor(0), boxes[:, 0]), torch.tensor(w - 1)) 
            boxes[:, 1] = torch.min(torch.max(torch.tensor(0), boxes[:, 1]), torch.tensor(h - 1))  
            boxes[:, 2] = torch.min(torch.max(torch.tensor(0), boxes[:, 2]), torch.tensor(w - 1))  
            boxes[:, 3] = torch.min(torch.max(torch.tensor(0), boxes[:, 3]), torch.tensor(h - 1))   

            mask = self.get_mask(boxes, 0, 0, w, h)
            boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
            labels = labels[mask.view(-1)]

            
            
            if boxes.size:  ## if all boxes are not expelled
                
                boxes4.append(boxes)
                labels4.append(labels)

        #plt.imshow(img4)
        #plt.show()

        img4 = img4[yc : yc + h:, xc : xc + w]
        #plt.imshow(img4)
        #plt.show()

        boxes4 = torch.cat(boxes4, 0)
        labels4 = torch.cat(labels4, 0)

        #
        return img4, boxes4, labels4

    def xywhn2xyxy(self, x, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] + padw  # top left x
        y[:, 1] = x[:, 1] + padh  # top left y
        y[:, 2] = x[:, 2] + padw  # bottom right x
        y[:, 3] = x[:, 3] + padh  # bottom right y
        
        return y

    def invalid_mask(self, boxes, wh_th = 2, area_th = 200, ar_th = 20):

        eps = 1e-9 

        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]      # width/height

        area = w * h                                                  # area
    
        ar = torch.max(w / (h + eps), h / (w + eps))              # aspect ratio

        mask = (w > wh_th) & (h > wh_th) & (area > area_th) & (ar < ar_th)

        return mask.view(-1, 1)

    def get_mask(self, boxes, x1, y1, x2, y2):

        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        
        x, y, w, h = x1, y1, x2 - x1, y2 - y1        ## correction made june 19
        #
        center = center - torch.FloatTensor([[x, y]]).expand_as(center)
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)

        mask = (mask1 & mask2).view(-1, 1)
        #
        return mask

    #################################################################################################################
    #################################################################################################################
    '''helper functions'''
    def bbox_wh_iou(self, wh1, wh2):
        
        #wh2 = wh2.t()
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1 + self.epsilon) + w2 * h2 - inter_area

        return inter_area / union_area


    def _encoder(self, boxes, labels, anchors, grid_num):

        stride = self.image_size // grid_num
        anchors = torch.tensor(anchors) / stride           # normalized anchors

        num_anchors = len(anchors)
        bbox_attrs = 5 + self.num_classes

        '''target 3 dimensional'''
        target = torch.FloatTensor(num_anchors, grid_num, grid_num, bbox_attrs).fill_(0)
        
        object_mask = torch.BoolTensor(num_anchors, grid_num, grid_num, bbox_attrs).fill_(False)       # should be expanded to all channels and anchor dimensions
        no_object_mask = torch.BoolTensor(num_anchors, grid_num, grid_num, bbox_attrs).fill_(True) 
        
        #
        #print(boxes)
        '''find the center coordinate of boxes and their width and height

           and also normalize by network stride

        '''
        '''
        variable boxes should be a tensor
        '''
        wh = boxes[:, 2:] / stride - boxes[:, :2] / stride
        cxcy = (boxes[:, 2:] / stride + boxes[:, :2] / stride) / 2
        #
        '''the center of the bounding boxes will be encoded as the offset from the nearest grid cell location
           the width and height will be encoded as factors of the width abnd height of the nearest anchor box (with highest IoU) 

        '''
        for i in range(cxcy.size()[0]):

            '''instead of IoU use the ratio of width or height'''

            '''from ultralytics yolov5 Glenn Jocher'''
            r = torch.stack([wh[i] / anchor for anchor in anchors])
            r = torch.max(r, 1. / r).max(dim = 1)[0] < self.anchor_t

            cxcy_sample = cxcy[i]
            ij = cxcy_sample.floor() #

            
            object_mask[r, int(ij[1]), int(ij[0]), :] = True 
            no_object_mask[r, int(ij[1]), int(ij[0]), :] = False

            target[r, int(ij[1]), int(ij[0]), 4] = 1
            target[r, int(ij[1]), int(ij[0]), int(labels[i]) + 4] = 1
                    
            target[r, int(ij[1]), int(ij[0]), 2:4] = torch.log(wh[i] / anchors[r] + self.epsilon)
            target[r, int(ij[1]), int(ij[0]), :2] = (cxcy_sample - ij) / self.alpha + (self.alpha - 1.0) / (2 * self.alpha)

            '''    
            object_mask[best_n, int(ij[1]), int(ij[0]), :] = True 
            no_object_mask[best_n, int(ij[1]), int(ij[0]), :] = False
            
            target[best_n, int(ij[1]), int(ij[0]), 4] = 1
            target[best_n, int(ij[1]), int(ij[0]), int(labels[i]) + 4] = 1
            
            target[best_n, int(ij[1]), int(ij[0]), 2:4] = torch.log(wh[i] / anchors[best_n] + self.epsilon)
            target[best_n, int(ij[1]), int(ij[0]), :2] = (cxcy_sample - ij) / self.alpha + (self.alpha - 1.0) / (2 * self.alpha)
            '''

            
            #
        
        
        return target, object_mask, no_object_mask


    def _encoder_anchorless(self, boxes, labels, level, grid_num):


        stride = self.image_size // grid_num
       
        bbox_attrs = 5 + self.num_classes

        '''target 3 dimensional'''
        target = torch.FloatTensor(1, grid_num, grid_num, bbox_attrs).fill_(0)
        
        object_mask = torch.BoolTensor(1, grid_num, grid_num, bbox_attrs).fill_(False)       # should be expanded to all channels and anchor dimensions
        no_object_mask = torch.BoolTensor(1, grid_num, grid_num, bbox_attrs).fill_(True) 
        
        #
        '''find the center coordinate of boxes and their width and height

           and also normalize by network stride

        '''
        '''
        variable boxes should be a tensor
        '''
        xy_1 = boxes[:, :2] / stride
        xy_2 = boxes[:, 2:] / stride
        wh = boxes[:, 2:] / stride - boxes[:, :2] / stride
        cxcy = (boxes[:, 2:] / stride + boxes[:, :2] / stride) / 2
        #
        '''the center of the bounding boxes will be encoded as the offset from the nearest grid cell location
           the width and height will be encoded as factors of the width abnd height of the nearest anchor box (with highest IoU) 

        '''
        for i in range(cxcy.size()[0]):

            #print('********box start********')

            cxcy_sample = cxcy[i]
            ij = cxcy_sample.floor() #


            if torch.max(wh[i]) > self.size_range[level] / stride:

                object_mask[0, int(ij[1]), int(ij[0]), :] = True 
                no_object_mask[0, int(ij[1]), int(ij[0]), :] = False

                
                target[0, int(ij[1]), int(ij[0]), 4] = 1
                target[0, int(ij[1]), int(ij[0]), int(labels[i]) + 4] = 1
                
                target[0, int(ij[1]), int(ij[0]), 2:4] = torch.log(wh[i] + self.epsilon)
                target[0, int(ij[1]), int(ij[0]), :2] = cxcy_sample - ij
                
                #
        
        
        return target, object_mask, no_object_mask



    def encoder(self, boxes, labels):
        
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        
        '''

        # anchor is either a list of two lists or a list of tuples
        
        # when list of lists


        grid_num = [self.image_size // self.stride, 
                    2 * self.image_size // self.stride, 
                    4 * self.image_size // self.stride
                    ]
        

        target, mask_obj, mask_no_obj = [], [], []      # list of tensor objects

        if self.anchors:

            for anchors, grid in zip(self.anchors, grid_num):

                target_, mask_obj_, mask_no_obj_ = self._encoder(boxes, labels, anchors = anchors, grid_num = grid)

                target.append(target_)
                mask_obj.append(mask_obj_)
                mask_no_obj.append(mask_no_obj_)

        else:

            for i, grid in enumerate(grid_num):
                target_, mask_obj_, mask_no_obj_ = self._encoder_anchorless(boxes, labels, level = i, grid_num = grid)

                target.append(target_)
                mask_obj.append(mask_obj_)
                mask_no_obj.append(mask_no_obj_)
        
        return target, mask_obj, mask_no_obj
    
    #
    def BGR2RGB(self,img):
        '''cv2 reads BGR - convert to RGB'''
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    def BGR2HSV(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    def HSV2BGR(self,img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    #
    def BGR2YUV(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #
    def YUV2BGR(self,img):
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    #
    def hsv_augment(self, bgr):

        dtype = bgr.dtype  # uint8
        hsv = self.BGR2HSV(bgr)
        hue, sat, val = cv2.split(hsv)

        #https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        #r = np.random.uniform(-1, 1, 3) * [0.5, 0.5, 0.5] + 1  # random gains
        r = np.random.uniform(-1, 1, 3) * [0.02, 0.7, 0.4] + 1  # random gains
        
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        bgr = self.HSV2BGR(hsv)

        #
        return bgr
    
    #
    def hist_equalize(self, bgr, clahe = True):
        #https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
        yuv = self.BGR2YUV(bgr)
        
        if clahe:  # contrast-limited adaptive histogram equalization
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
        #
        bgr = self.YUV2BGR(yuv)
        
        return bgr

    #
    def randomBlur(self,bgr):
        #
        if random.random() < 0.5:
            k = 2 * random.randint(1, 7) + 1
            bgr = cv2.blur(bgr,(k, k))
        #
        return bgr
    #
    def randomShift(self, bgr, boxes, labels):
        
        bgr_in = bgr.copy()
        boxes_in = boxes.detach().clone()
        labels_in = labels.detach().clone()

        '''image diemsnions'''
        height, width, _ = bgr.shape
        
        ''' translation matrix'''
        shift_x = random.uniform(-width * 0.2, width * 0.2)
        shift_y = random.uniform(-height * 0.2, height * 0.2)
        #
        M = np.eye(2, 3)
        M[0, 2] = shift_x
        M[1, 2] = shift_y

        '''apply translation'''
        border_value = self.mean
        border_value = border_value[::-1] 

        bgr = cv2.warpAffine(bgr, M, (width, height), borderValue = border_value)    # keep the dimension of the original
                                                                                     # assign boder values to data set mean

        '''now translate the boxes and labels
           exclude boxes and labels that go out of bound
        '''

        '''first, convert boxes to numpy'''
        boxes = boxes.reshape((-1, 2))                                                    # N x 4 -> 2N x 2
        boxes = boxes.detach().numpy()
        boxes = np.hstack([boxes, np.ones((len(boxes), 1))])                         # concate 1s to (x, y) coordinates

        # transform
        boxes = M.dot(boxes.T).T                                                      
        boxes = np.clip(boxes, np.array([0., width]), np.array([0., height]))        # clip out of bound boxes
                                                                                     # some boxes will completely go out  
                                                                                     # need to delete them & their corresponding labels
        '''back to torch tensor;
           and apply a mask to exclude invalid boxes & labels
        '''
        boxes = torch.from_numpy(boxes)                                              # back to torch
        boxes = boxes.reshape((-1, 4))                                                   # 2N x 2 -> N x 4

        '''box centers '''
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

        mask1 = (cxcy[:, 0] > 0) & (cxcy[:, 0] < width)
        mask2 = (cxcy[:, 1] > 0) & (cxcy[:, 1] < height)
        mask = (mask1 & mask2).view(-1,1)                                            # mask to retain

        boxes = boxes[mask.expand_as(boxes)].view(-1,4)                              # retained boxes
        labels = labels[mask.view(-1)]                                               # retained labels

        if len(boxes) == 0:
            '''if the shift ends up losing all the b-boxes,
                   discard it and return the unchanged image and labels
            '''
            return bgr_in, boxes_in, labels_in
        
        else:

            return bgr, boxes, labels

    #
    def randomZoom(self, bgr, boxes, labels, is_mosaic = False):
        #

        bgr_in = bgr.copy()
        boxes_in = boxes.detach().clone()
        labels_in = labels.detach().clone()

        '''image diemsnions'''
        height, width, _ = bgr.shape

        '''transformation matrix'''
        if is_mosaic:
            scale = random.uniform(0.3, 1.5)
            bgr_ctr = (int(width / 2 + random.uniform(-200, 200)), int(height / 2 + random.uniform(-200, 200)))

        else:
            scale = random.uniform(0.1, 1.5)
            bgr_ctr = (int(width / 2 + random.uniform(-50, 50)), int(height / 2 + random.uniform(-50, 50)))

        M = cv2.getRotationMatrix2D(angle = 0, center = bgr_ctr, scale = scale)

        '''apply transformation'''
        border_value = self.mean
        border_value = border_value[::-1] 

        bgr = cv2.warpAffine(bgr, M, (width, height), borderValue = border_value)    # keep the dimension of the original
                                                                                     # assign boder values to data set mean

        '''now translate the boxes and labels
           exclude boxes and labels that go out of bound
        '''

        '''first, convert boxes to numpy'''
        boxes = boxes.reshape((-1, 2))                                                  # N x 4 -> 2N x 2
        boxes = boxes.detach().numpy()
        boxes = np.hstack([boxes, np.ones((len(boxes), 1))])                         # concate 1s to (x, y) coordinates

        # transform
        boxes = M.dot(boxes.T).T                                                      
        boxes = np.clip(boxes, np.array([0., width]), np.array([0., height]))        # clip out of bound boxes
                                                                                     # some boxes will completely go out  
                                                                                     # need to delete them & their corresponding labels
        '''back to torch tensor;
           and apply a mask to exclude invalid boxes & labels
        '''
        boxes = torch.from_numpy(boxes)                                              # back to torch
        boxes = boxes.reshape((-1, 4))                                                    # 2N x 2 -> N x 4

        '''box centers '''
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

        mask1 = (cxcy[:, 0] > 0) & (cxcy[:, 0] < width)
        mask2 = (cxcy[:, 1] > 0) & (cxcy[:, 1] < height)
        mask = (mask1 & mask2).view(-1,1)                                            # mask to retain

        boxes = boxes[mask.expand_as(boxes)].view(-1,4)                              # retained boxes
        labels = labels[mask.view(-1)]                                               # retained labels

        if len(boxes) == 0:
            '''if the shift ends up losing all the b-boxes,
                   discard it and return the unchanged image and labels
            '''
            return bgr_in, boxes_in, labels_in
        
        else:

            return bgr, boxes, labels
    

    def resizeMosaic(self, rgb, boxes):
        #
        '''find the ratio'''
        old_size = rgb.shape[:2]
        desired_size = self.image_size

        ratio = 1
        #
        if not max(old_size) == self.image_size:
  
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            #
            '''now resize the image'''
            rgb = cv2.resize(rgb, (new_size[1], new_size[0]))                               # remember: resize takes (W, H)
            #
        #
        #
        '''do not forget to resize the b-boxes'''  
        #print(ratio)                                  
        scale_tensor = torch.FloatTensor([[ratio, ratio, ratio, ratio]]).expand_as(boxes)
        boxes = boxes * scale_tensor
        #
        return rgb, boxes    

    #
    '''method for image resize to self.image_size
       https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    '''
    def Resize(self, rgb, boxes):
        #
        '''find the ratio'''
        old_size = rgb.shape[:2]
        desired_size = self.image_size
        #
        if max(old_size) < self.image_size:
            delta_w = desired_size - old_size[1]
            delta_h = desired_size - old_size[0]
            ratio = 1.0
        else:
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            #
            '''now resize the image'''
            rgb = cv2.resize(rgb, (new_size[1], new_size[0]))                               # remember: resize takes (W, H)
            #
            '''now find the padding size '''
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
        #

        if self.train:
            a, b = random.uniform(0, 1.0),  random.uniform(0, 1.0)
            top = int(a * delta_h)
            bottom = delta_h - top
            left = int(b * delta_w)
            right = delta_w - left
        else:
            top, bottom = 0, delta_h
            left, right = 0, delta_w
        
        #top, bottom = 0, delta_h
        #left, right = 0, delta_w
        
        #
        '''set the padding value to data mean'''
        color = self.mean
        color = color[::-1] 
        #
        '''now, do the padding'''
        rgb_pad = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)
        #
        '''do not forget to resize the b-boxes'''  
        #print(ratio)
        box_shift = torch.FloatTensor([[left, top, left, top]]).expand_as(boxes)                                     
        scale_tensor = torch.FloatTensor([[ratio, ratio, ratio, ratio]]).expand_as(boxes)
        boxes = boxes * scale_tensor + box_shift
        #boxes = boxes * scale_tensor
        #
        return rgb_pad, boxes
    #
    def random_flip(self, im, boxes):
        #
        '''horizontal flip'''
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            #
            im = im_lr
        #
        return im, boxes



class FusionDataset(data.Dataset):
    #
    
    #######
    def __init__(self, root, list_file, image_size, anchors, num_classes = 1, alpha = 1.0, stack = False, train = True, transform = None):
        #
        print('data init')
        self.image_size = image_size       # training image size
        self.mosaic_border = [-image_size // 2, -image_size // 2]
        self.anchors = anchors             # anchor boxes
        self.num_classes = num_classes
        self.stride = 32
        self.root = root                   # root directory of the jpg images
        self.train = train                 # bool to indicate train or validation
        self.transform = transform
        self.fnames = []                   # all the training file names accumulated here
        self.boxes = []
        self.labels = []
        self.epsilon = 1e-16
        self.rgb_mean = (151, 152, 148)        # data set mean for RGB channels
                                           # these values will be used for padding
                                           # and eventually after normalization the padding 
                                           # values will be zero
        
        
        self.ir_mean = (135, 135, 135)
        
        self.stack = stack

        self.alpha = alpha
        #self.size_weights = [3.0, 2.0, 1.0]


        #
        '''read all the lines'''
        with open(list_file) as f:         # list_file is the annotation text file for the images in root directory
            lines  = f.readlines()
        #
        for line in lines:

            if line.startswith('#'):
                continue
            splited = line.strip().split()
            if not splited:
                continue
            #
            name, ext = splited[0].split('.')
            rgb_name, ir_name = name, name
            rgb_ext, ir_ext = 'jpg', 'jpeg'
            
            rgb_name = rgb_name.split('\\')
            rgb_name.insert(1, 'registered_may')
            rgb_name = '\\'.join(rgb_name)
            rgb_name = rgb_name + '.' + rgb_ext

            ir_name = ir_name.split('\\')
            ir_name.insert(1, 'registered_may')
            ir_name = '\\'.join(ir_name)
            ir_name = ir_name + '.' + ir_ext
            #
            self.fnames.append((rgb_name, ir_name))

            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                '''coordinates'''
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 +  5* i])
                y2 = float(splited[4 + 5 * i])
                '''class'''
                c = splited[5 + 5 * i]
                box.append([x, y, x2, y2])
                label.append(int(c) + 1)
            #
            self.boxes.append(torch.Tensor(box))           # float 32
            self.labels.append(torch.LongTensor(label))    # int 64
        #
        self.num_samples = len(self.boxes)                 # total number of annotated bounding boxes
        self.indices = range(len(self.boxes))              # indices of the imagers which will be later used in getitem() 
                                                           # and mosaicing
    #########
    def __getitem__(self, idx):
        #
        '''this is the method for retrieving data with the iterator'''
        #
        index = self.indices[idx]
        #print(self.fnames[index])
        #print(index)
        
        if self.train:                                     # data augmentation only during training
            ''' all these are done in BGR'''
            
            '''image-depended mosaicing'''

            #boxes = self.boxes[index].clone()
            #mosaic_prob = 1.0 - self.image_weight_from_size(boxes)
            mosaic_prob = 0.4

            mosaic = random.random() < mosaic_prob

            if mosaic: 
                rgb, ir, boxes, labels = self.load_mosaic_4(index)  # composite image

                #print(rgb.shape, ir.shape)

                
                if random.random() < 0.1:
                    rgb2, ir2, boxes2, labels2 = self.load_mosaic_4(random.randint(0, self.num_samples - 1))
                    r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                    rgb = (rgb * r + rgb2 * (1 - r)).astype(np.uint8)
                    ir = (ir * r + ir2 * (1 - r)).astype(np.uint8)
                    labels = torch.cat((labels, labels2), dim = 0)
                    boxes = torch.cat((boxes, boxes2), dim = 0)

                

            else:
                rgb, ir, _, boxes, labels = self.load_image(index)  # single image 

                
                if random.random() < 0.1:
                    rgb2, ir2, boxes2, labels2 = self.load_mosaic_4(random.randint(0, self.num_samples - 1))
                    r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                    rgb = (rgb * r + rgb2 * (1 - r)).astype(np.uint8)
                    ir = (ir * r + ir2 * (1 - r)).astype(np.uint8)
                    labels = torch.cat((labels, labels2), dim = 0)
                    boxes = torch.cat((boxes, boxes2), dim = 0)
                
            
            rgb, ir, boxes, labels = self.randomShift(rgb,ir,boxes,labels)
            rgb, ir, boxes, labels = self.randomZoom(rgb,ir, boxes, labels, is_mosaic = mosaic)
            
            rgb, ir, boxes = self.random_flip(rgb,ir, boxes)
            rgb = self.randomBlur(rgb)
            ir = self.randomBlur(ir)

            rgb = self.hsv_augment(rgb)
            ir = self.hsv_augment(ir)

            if random.random() < 0.5:
                rgb = self.hist_equalize(rgb)

            if random.random() < 0.5:
                ir = self.hist_equalize(ir)

        #

        else:   # test/val

            rgb, ir, _, boxes, labels = self.load_image(index)  # single image with no augmentation
            
        '''resize to network dimension
        '''
        rgb, ir, boxes = self.Resize(rgb, ir, boxes)

        '''
        convert BGR to RGB, since pret-trained models use RGB
        '''
        
        rgb = self.BGR2RGB(rgb)
        ir = self.BGR2RGB(ir)
        
        '''
        fix b-box size if required
        '''
        boxes = boxes.clamp(0, self.image_size)            # keeping the box within the bound of the image
        
        '''important  !!!!!!!
           unusual boxes, such as, less than 2 pixels in either dimension,
           or unusually narrow or long, are excluded
        '''
        invalid = self.invalid_mask(boxes)
        boxes = boxes[invalid.expand_as(boxes)].view(-1, 4)
        labels = labels[invalid.view(-1)]

        
        '''
        weights = torch.zeros((boxes.shape[0], 1))
        for i, box in enumerate(boxes):

            area = (box[2] - box[0]) * (box[3] - box[1])
            
            if  area <= 32 ** 2:
                weights[i] = 3.0

            if  area > 32 ** 2 and area <= 96 ** 2:
                weights[i] = 2.0

            if  area > 96 ** 2:
                weights[i] = 1.0

        boxes = torch.cat([boxes, weights], dim = 1)
        '''

        ##### encoding


        #target, mask_obj, box_weight  = self.encoder(boxes, labels)
        target, mask_obj, mask_no_obj = self.encoder(boxes, labels)

        #print(boxes.shape)
        #print(labels.unsqueeze(dim = 1).shape)

        boxes = torch.cat((boxes, labels.float().unsqueeze(dim = 1) - 1.0), dim = 1)
        gt_boxes = torch.zeros((len(boxes), 6))
        gt_boxes[:, 1:] = boxes
        #
        if self.transform:
            rgb_n = self.transform(rgb)
            ir_n = self.transform(ir)

        else:
            '''very important: rgb uas shape [w, h, ch]
                
            '''
            rgb_n = torch.tensor(rgb.copy()).permute(2, 0, 1)
            ir_n = torch.tensor(ir.copy()).permute(2, 0, 1)
        #
        #return img, boxes, labels, img_n, target
        return gt_boxes, rgb_n, ir_n, target, mask_obj, mask_no_obj

    def collate_fn(self, batch):

        boxes, rgbs, irs, targets, mask_obj, mask_no_obj = list(zip(*batch))
        
        # Remove empty placeholder targets
        gt_boxes = [box for box in boxes if box is not None]
        # Add sample index to targets
        for i, box in enumerate(gt_boxes):
            box[:, 0] = i
        gt_boxes = torch.cat(gt_boxes, dim = 0)


        rgbs = torch.stack(rgbs)
        irs = torch.stack(irs)

        '''targets: a list of lists
           number of lists inside equals batch_size,
           each list has exactly two tensors (13 x 13 & 26 x 26)

        '''
        targets = list(zip(*targets))
        mask_obj = list(zip(*mask_obj))
        mask_no_obj = list(zip(*mask_no_obj))

        targets = [torch.stack(item) for item in targets]
        mask_obj = [torch.stack(item) for item in mask_obj]
        mask_no_obj = [torch.stack(item) for item in mask_no_obj]

        
        #return gt_boxes, imgs, targets, mask_obj, mask_no_obj
        if self.stack:
            imgs = torch.cat((rgbs, irs), dim = 1)
            return imgs, gt_boxes, targets, mask_obj, mask_no_obj
        else:
            return rgbs, irs, gt_boxes, targets, mask_obj, mask_no_obj

    ###########
    def __len__(self):
        return self.num_samples
    
    #
    def load_image(self, idx):

        rgb_name, ir_name = self.fnames[idx]
        '''read image (BGR) and b-boxes'''
        rgb = cv2.imread(os.path.join(self.root, rgb_name))
        ir = cv2.imread(os.path.join(self.root, ir_name))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        #
        return rgb, ir, rgb.shape[:2], boxes, labels
    #

    def image_weight_from_size(self, boxes):
    
        weights = torch.zeros((boxes.shape[0], 1))
        for i, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
                
            if  area <= 32 ** 2:
                weights[i] = 0.5

            if  area > 32 ** 2 and area <= 96 ** 2:
                weights[i] = 1.0 / 3.0

            if  area > 96 ** 2:
                weights[i] = 1.0 / 6.0

        return weights.mean().item()

    #

    def load_mosaic(self, idx):

        rgb_mean = self.rgb_mean
        rgb_mean = rgb_mean[::-1] 

        ir_mean = self.ir_mean
        ir_mean = ir_mean[::-1]
        
        boxes4, labels4 = [], []
        s = self.image_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [idx] + [self.indices[random.randint(0, self.num_samples - 1)] for _ in range(3)]  # 3 additional image indices

        
        for i, index in enumerate(indices):
            # Load image
            rgb, ir, _, boxes, labels  = self.load_image(index)
            #img, boxes = self.resizeMosaic(img, boxes)
            h, w = rgb.shape[:2]

            # place img in img4
            if i == 0:  # top left
                rgb4 = np.full((s * 2, s * 2, rgb.shape[2]), rgb_mean, dtype=np.uint8)  # base image with 4 tiles
                ir4 = np.full((s * 2, s * 2, ir.shape[2]), ir_mean, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            rgb4[y1a:y2a, x1a:x2a] = rgb[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            ir4[y1a:y2a, x1a:x2a] = ir[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            #
            #mask = self.get_mask(boxes, x1b, y1b, x2b, y2b)
            #boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
            #labels = labels[mask.view(-1)]
            #
            if boxes.size:  ## if all boxes are not expelled
                boxes = self.xywhn2xyxy(boxes, padw, padh) 

                boxes4.append(boxes)
                labels4.append(labels)

        boxes4 = torch.cat(boxes4, 0)
        labels4 = torch.cat(labels4, 0)

        #boxes4 = boxes4.clamp(0, 2 * s)

        
        #
        return rgb4, ir4, boxes4, labels4

    def load_mosaic_4(self, idx):

        rgb_mean = self.rgb_mean
        rgb_mean = rgb_mean[::-1] 

        ir_mean = self.ir_mean
        ir_mean = ir_mean[::-1]
        
        boxes4, labels4 = [], []
        
        yc = int(random.uniform(0, 512))
        xc = int(random.uniform(0, 640))  # mosaic center x, y
        #print(xc)
        indices = [idx] + [self.indices[random.randint(0, self.num_samples - 1)] for _ in range(3)]  # 3 additional image indices

        for i, index in enumerate(indices):
            # Load image
            rgb, ir, _, boxes, labels  = self.load_image(index)
            #img, boxes = self.resizeMosaic(img, boxes)
            h, w = rgb.shape[:2]

            # place img in img4
            if i == 0:  # top left
                rgb4 = np.full((h * 2, w * 2, rgb.shape[2]), rgb_mean, dtype=np.uint8)  # base image with 2 tiles
                ir4 = np.full((h * 2, w * 2, ir.shape[2]), ir_mean, dtype=np.uint8)  # base image with 4 tiles

                rgb4[: h, : w] = rgb
                ir4[: h, : w] = ir

                boxes[:, 0] = boxes[:, 0] + 0 - xc
                boxes[:, 1] = boxes[:, 1] + 0 - yc 
                boxes[:, 2] = boxes[:, 2] + 0 - xc
                boxes[:, 3] = boxes[:, 3] + 0 - yc


            elif i == 1:  # top right

                rgb4[: h, w : ] = rgb
                ir4[: h, w : ] = ir

                boxes[:, 0] = boxes[:, 0] + w - xc
                boxes[:, 1] = boxes[:, 1] + 0 - yc
                boxes[:, 2] = boxes[:, 2] + w - xc
                boxes[:, 3] = boxes[:, 3] + 0 - yc


            elif i == 2:  # bottom left

                rgb4[h :, : w] = rgb
                ir4[h :, : w] = ir

                boxes[:, 0] = boxes[:, 0] + 0 - xc
                boxes[:, 1] = boxes[:, 1] + h - yc
                boxes[:, 2] = boxes[:, 2] + 0 - xc
                boxes[:, 3] = boxes[:, 3] + h - yc


            elif i == 3:  # bottom right

                rgb4[h :, w : ] = rgb
                ir4[h :, w : ] = ir

                boxes[:, 0] = boxes[:, 0] + w - xc
                boxes[:, 1] = boxes[:, 1] + h - yc
                boxes[:, 2] = boxes[:, 2] + w - xc
                boxes[:, 3] = boxes[:, 3] + h - yc

            
            boxes[:, 0] = torch.min(torch.max(torch.tensor(0), boxes[:, 0]), torch.tensor(w - 1)) 
            boxes[:, 1] = torch.min(torch.max(torch.tensor(0), boxes[:, 1]), torch.tensor(h - 1))  
            boxes[:, 2] = torch.min(torch.max(torch.tensor(0), boxes[:, 2]), torch.tensor(w - 1))  
            boxes[:, 3] = torch.min(torch.max(torch.tensor(0), boxes[:, 3]), torch.tensor(h - 1))   

            mask = self.get_mask(boxes, 0, 0, w, h)
            boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
            labels = labels[mask.view(-1)]

            
            
            if boxes.size:  ## if all boxes are not expelled
                
                boxes4.append(boxes)
                labels4.append(labels)

        #plt.imshow(img4)
        #plt.show()

        rgb4 = rgb4[yc : yc + h:, xc : xc + w]
        ir4 = ir4[yc : yc + h:, xc : xc + w]
        #plt.imshow(img4)
        #plt.show()

        boxes4 = torch.cat(boxes4, 0)
        labels4 = torch.cat(labels4, 0)

        #
        return rgb4, ir4, boxes4, labels4

    def xywhn2xyxy(self, x, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] + padw  # top left x
        y[:, 1] = x[:, 1] + padh  # top left y
        y[:, 2] = x[:, 2] + padw  # bottom right x
        y[:, 3] = x[:, 3] + padh  # bottom right y
        
        return y

    def invalid_mask(self, boxes, wh_th = 2, area_th = 200, ar_th = 20):

        eps = 1e-9 

        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]      # width/height

        area = w * h                                                  # area
    
        ar = torch.max(w / (h + eps), h / (w + eps))              # aspect ratio

        mask = (w > wh_th) & (h > wh_th) & (area > area_th) & (ar < ar_th)

        return mask.view(-1, 1)

    def get_mask(self, boxes, x1, y1, x2, y2):

        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        #
        center = center - torch.FloatTensor([[x, y]]).expand_as(center)
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)

        mask = (mask1 & mask2).view(-1, 1)
        #
        return mask

    #################################################################################################################
    #################################################################################################################
    '''helper functions'''
    def bbox_wh_iou(self, wh1, wh2):
        
        #wh2 = wh2.t()
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1 + self.epsilon) + w2 * h2 - inter_area

        return inter_area / union_area


    def _encoder(self, boxes, labels, anchors, grid_num):

        stride = self.image_size // grid_num
        anchors = torch.tensor(anchors) / stride           # normalized anchors

        num_anchors = len(anchors)
        bbox_attrs = 5 + self.num_classes

        '''target 3 dimensional'''
        target = torch.FloatTensor(num_anchors, grid_num, grid_num, bbox_attrs).fill_(0)
        
        object_mask = torch.BoolTensor(num_anchors, grid_num, grid_num, bbox_attrs).fill_(False)       # should be expanded to all channels and anchor dimensions
        no_object_mask = torch.BoolTensor(num_anchors, grid_num, grid_num, bbox_attrs).fill_(True) 
        
        #
        #print(boxes)
        '''find the center coordinate of boxes and their width and height

           and also normalize by network stride

        '''
        '''
        variable boxes should be a tensor
        '''
        wh = boxes[:, 2:] / stride - boxes[:, :2] / stride
        cxcy = (boxes[:, 2:] / stride + boxes[:, :2] / stride) / 2
        #
        '''the center of the bounding boxes will be encoded as the offset from the nearest grid cell location
           the width and height will be encoded as factors of the width abnd height of the nearest anchor box (with highest IoU) 

        '''
        for i in range(cxcy.size()[0]):

            '''first calculate the IoU of the bounding box with each anchor box and pick the one with largest IoU'''

            ious = torch.stack([self.bbox_wh_iou(anchor, wh[i]) for anchor in anchors])
            best_iou, best_n = ious.max(dim = 0)

            cxcy_sample = cxcy[i]
            ij = cxcy_sample.floor() #

            object_mask[best_n, int(ij[1]), int(ij[0]), :] = True 
            no_object_mask[best_n, int(ij[1]), int(ij[0]), :] = False

            
            target[best_n, int(ij[1]), int(ij[0]), 4] = 1
            target[best_n, int(ij[1]), int(ij[0]), int(labels[i]) + 4] = 1
            
            target[best_n, int(ij[1]), int(ij[0]), 2:4] = torch.log(wh[i] / anchors[best_n] + self.epsilon)
            target[best_n, int(ij[1]), int(ij[0]), :2] = (cxcy_sample - ij) / self.alpha + (self.alpha - 1.0) / (2 * self.alpha)


            
            #
        
        
        return target, object_mask, no_object_mask



    def encoder(self, boxes, labels):
        
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        
        '''

        # anchor is either a list of two lists or a list of tuples
        
        # when list of lists

        self.image_size // self.stride

        if len(self.anchors) == 3:
            grid_num = [self.image_size // self.stride, 
                        2 * self.image_size // self.stride, 
                        4 * self.image_size // self.stride
                       ]
        
        elif len(self.anchors) == 2:
            grid_num = [self.image_size // self.stride, 
                        2 * self.image_size // self.stride, 
                       ]
        
        # otherwise
        else:
            grid_num = self.image_size // self.stride

        
        if not isinstance(grid_num, list):

            target, mask_obj, mask_no_obj = self._encoder(boxes, labels, anchors = self.anchors, grid_num = grid_num)                                      # 7x7x30


        else:

            target, mask_obj, mask_no_obj = [], [], []      # list of tensor objects

            for anchors, grid in zip(self.anchors, grid_num):

                target_, mask_obj_, mask_no_obj_ = self._encoder(boxes, labels, anchors = anchors, grid_num = grid)

                target.append(target_)
                mask_obj.append(mask_obj_)
                mask_no_obj.append(mask_no_obj_)
        
        return target, mask_obj, mask_no_obj
    
    #
    def BGR2RGB(self,img):
        '''cv2 reads BGR - convert to RGB'''
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    def BGR2HSV(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    def HSV2BGR(self,img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    #
    def BGR2YUV(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #
    def YUV2BGR(self,img):
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    #
    def hsv_augment(self, bgr):

        dtype = bgr.dtype  # uint8
        hsv = self.BGR2HSV(bgr)
        hue, sat, val = cv2.split(hsv)

        #https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        #r = np.random.uniform(-1, 1, 3) * [0.5, 0.5, 0.5] + 1  # random gains
        r = np.random.uniform(-1, 1, 3) * [0.02, 0.7, 0.4] + 1  # random gains
        
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        bgr = self.HSV2BGR(hsv)

        #
        return bgr
    
    #
    def hist_equalize(self, bgr, clahe = True):
        #https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
        yuv = self.BGR2YUV(bgr)
        
        if clahe:  # contrast-limited adaptive histogram equalization
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
        #
        bgr = self.YUV2BGR(yuv)
        
        return bgr

    #
    def randomBlur(self,bgr):
        #
        if random.random() < 0.5:
            k = 2 * random.randint(1, 7) + 1
            bgr = cv2.blur(bgr,(k, k))
        #
        return bgr
    #
    
    def randomShift(self, bgr, ir, boxes, labels):


        bgr_in = bgr.copy()
        ir_in = ir.copy()

        boxes_in = boxes.detach().clone()
        labels_in = labels.detach().clone()

        '''image diemsnions'''
        height, width, _ = bgr.shape
        
        ''' translation matrix'''
        shift_x = random.uniform(-width * 0.2, width * 0.2)
        shift_y = random.uniform(-height * 0.2, height * 0.2)
        #
        M = np.eye(2, 3)
        M[0, 2] = shift_x
        M[1, 2] = shift_y

        '''apply translation'''
        border_value = self.rgb_mean
        border_value = border_value[::-1] 

        bgr = cv2.warpAffine(bgr, M, (width, height), borderValue = border_value)    # keep the dimension of the original
                                                                                     # assign boder values to data set mean

        border_value = self.ir_mean
        border_value = border_value[::-1] 

        ir = cv2.warpAffine(ir, M, (width, height), borderValue = border_value)    # keep the dimension of the original
                                                                                     # assign boder values to data set mean

        '''now translate the boxes and labels
           exclude boxes and labels that go out of bound
        '''

        '''first, convert boxes to numpy'''
        boxes = boxes.reshape((-1, 2))                                                    # N x 4 -> 2N x 2
        boxes = boxes.detach().numpy()
        boxes = np.hstack([boxes, np.ones((len(boxes), 1))])                         # concate 1s to (x, y) coordinates

        # transform
        boxes = M.dot(boxes.T).T                                                      
        boxes = np.clip(boxes, np.array([0., width]), np.array([0., height]))        # clip out of bound boxes
                                                                                     # some boxes will completely go out  
                                                                                     # need to delete them & their corresponding labels
        '''back to torch tensor;
           and apply a mask to exclude invalid boxes & labels
        '''
        boxes = torch.from_numpy(boxes)                                              # back to torch
        boxes = boxes.reshape((-1, 4))                                                   # 2N x 2 -> N x 4

        '''box centers '''
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

        mask1 = (cxcy[:, 0] > 0) & (cxcy[:, 0] < width)
        mask2 = (cxcy[:, 1] > 0) & (cxcy[:, 1] < height)
        mask = (mask1 & mask2).view(-1,1)                                            # mask to retain

        boxes = boxes[mask.expand_as(boxes)].view(-1,4)                              # retained boxes
        labels = labels[mask.view(-1)]                                               # retained labels

        if len(boxes) == 0:
            '''if the shift ends up losing all the b-boxes,
                   discard it and return the unchanged image and labels
            '''
            return bgr_in, ir_in, boxes_in, labels_in
        
        else:

            return bgr, ir, boxes, labels

    #
    def randomZoom(self, bgr, ir, boxes, labels, is_mosaic = False):
        #

        bgr_in = bgr.copy()
        ir_in = ir.copy()

        boxes_in = boxes.detach().clone()
        labels_in = labels.detach().clone()

        '''image diemsnions'''
        height, width, _ = bgr.shape

        '''transformation matrix'''
        if is_mosaic:
            scale = random.uniform(0.3, 1.5)
            ctr = (int(width / 2 + random.uniform(-200, 200)), int(height / 2 + random.uniform(-200, 200)))

        else:
            scale = random.uniform(0.1, 1.5)
            ctr = (int(width / 2 + random.uniform(-50, 50)), int(height / 2 + random.uniform(-50, 50)))

        M = cv2.getRotationMatrix2D(angle = 0, center = ctr, scale = scale)

        '''apply transformation'''

        border_value = self.rgb_mean
        border_value = border_value[::-1] 

        bgr = cv2.warpAffine(bgr, M, (width, height), borderValue = border_value)    # keep the dimension of the original
                                                                                     # assign boder values to data set mean

        border_value = self.ir_mean
        border_value = border_value[::-1] 

        ir = cv2.warpAffine(ir, M, (width, height), borderValue = border_value)    # keep the dimension of the original
                                                                                     # assign boder values to data set mean

        '''now translate the boxes and labels
           exclude boxes and labels that go out of bound
        '''

        '''first, convert boxes to numpy'''
        boxes = boxes.reshape((-1, 2))                                                  # N x 4 -> 2N x 2
        boxes = boxes.detach().numpy()
        boxes = np.hstack([boxes, np.ones((len(boxes), 1))])                         # concate 1s to (x, y) coordinates

        # transform
        boxes = M.dot(boxes.T).T                                                      
        boxes = np.clip(boxes, np.array([0., width]), np.array([0., height]))        # clip out of bound boxes
                                                                                     # some boxes will completely go out  
                                                                                     # need to delete them & their corresponding labels
        '''back to torch tensor;
           and apply a mask to exclude invalid boxes & labels
        '''
        boxes = torch.from_numpy(boxes)                                              # back to torch
        boxes = boxes.reshape((-1, 4))                                                    # 2N x 2 -> N x 4

        '''box centers '''
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

        mask1 = (cxcy[:, 0] > 0) & (cxcy[:, 0] < width)
        mask2 = (cxcy[:, 1] > 0) & (cxcy[:, 1] < height)
        mask = (mask1 & mask2).view(-1,1)                                            # mask to retain

        boxes = boxes[mask.expand_as(boxes)].view(-1,4)                              # retained boxes
        labels = labels[mask.view(-1)]                                               # retained labels

        if len(boxes) == 0:
            '''if the shift ends up losing all the b-boxes,
                   discard it and return the unchanged image and labels
            '''
            return bgr_in, ir_in, boxes_in, labels_in
        
        else:

            return bgr, ir, boxes, labels
                                                    

    def resizeMosaic(self, rgb, boxes):
        #
        '''find the ratio'''
        old_size = rgb.shape[:2]
        desired_size = self.image_size

        ratio = 1
        #
        if not max(old_size) == self.image_size:
  
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            #
            '''now resize the image'''
            rgb = cv2.resize(rgb, (new_size[1], new_size[0]))                               # remember: resize takes (W, H)
            #
        #
        #
        '''do not forget to resize the b-boxes'''  
        #print(ratio)                                  
        scale_tensor = torch.FloatTensor([[ratio, ratio, ratio, ratio]]).expand_as(boxes)
        boxes = boxes * scale_tensor
        #
        return rgb, boxes    

    #
    '''method for image resize to self.image_size
       https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    '''
    def Resize(self, rgb, ir, boxes):
        #
        '''find the ratio'''
        old_size = rgb.shape[:2]
        desired_size = self.image_size
        #
        if max(old_size) < self.image_size:
            delta_w = desired_size - old_size[1]
            delta_h = desired_size - old_size[0]
            ratio = 1.0
        else:
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            #
            '''now resize the image'''
            rgb = cv2.resize(rgb, (new_size[1], new_size[0]))                               # remember: resize takes (W, H)
            ir = cv2.resize(ir, (new_size[1], new_size[0]))
            #
            '''now find the padding size '''
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
        #

        if self.train:
            a, b = random.uniform(0, 1.0),  random.uniform(0, 1.0)
            top = int(a * delta_h)
            bottom = delta_h - top
            left = int(b * delta_w)
            right = delta_w - left
        else:
            top, bottom = 0, delta_h
            left, right = 0, delta_w
        
        #top, bottom = 0, delta_h
        #left, right = 0, delta_w
        
        #
        '''set the padding value to data mean'''
        color = self.rgb_mean
        color = color[::-1] 
        #
        '''now, do the padding'''
        rgb_pad = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)

        color = self.ir_mean
        color = color[::-1] 
        #
        '''now, do the padding'''
        ir_pad = cv2.copyMakeBorder(ir, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)
        #
        '''do not forget to resize the b-boxes'''  
        #print(ratio)
        box_shift = torch.FloatTensor([[left, top, left, top]]).expand_as(boxes)                                     
        scale_tensor = torch.FloatTensor([[ratio, ratio, ratio, ratio]]).expand_as(boxes)
        boxes = boxes * scale_tensor + box_shift
        #boxes = boxes * scale_tensor
        #
        return rgb_pad, ir_pad, boxes
    #
    def random_flip(self, im1, im2, boxes):
        #
        '''horizontal flip'''
        if random.random() < 0.5:
            im1_lr = np.fliplr(im1).copy()
            im2_lr = np.fliplr(im2).copy()
            h, w, _ = im1.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            #
            im1 = im1_lr
            im2 = im2_lr
        #
        return im1, im2, boxes



# TODO: 1 . modularize all the augmentation functions 
class VisibleDataset(data.Dataset):
    #
    
    #######
    def __init__(self, root, list_file, image_size, anchors, num_classes = 1, alpha = 1.0, train = True, transform = None):
        #
        print('data init')
        self.image_size = image_size       # training image size
        self.mosaic_border = [-image_size // 2, -image_size // 2]
        self.anchors = anchors             # anchor boxes
        self.anchor_t = 4.
        self.num_classes = num_classes
        self.stride = 32
        self.root = root                   # root directory of the jpg images
        self.train = train                 # bool to indicate train or validation
        self.transform = transform
        self.fnames = []                   # all the training file names accumulated here
        self.boxes = []
        self.labels = []
        self.epsilon = 1e-16
        self.mean = (151, 152, 148)        # data set mean for RGB channels
                                           # these values will be used for padding
                                           # and eventually after normalization the padding 
                                           # values will be zero
        
        self.alpha = alpha
        #self.size_weights = [3.0, 2.0, 1.0]


        #
        '''read all the lines'''
        with open(list_file) as f:         # list_file is the annotation text file for the images in root directory
            lines  = f.readlines()
        #
        for line in lines:

            if line.startswith('#'):
                continue
            splited = line.strip().split()
            if not splited or len(splited) < 8:
                #print(splited[0], "no box")
                continue
            #
            name = splited[0]
            
            #
            #splited = line.strip().split()
            #self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 7
            box = []
            label = []
            for i in range(num_boxes):
                '''coordinates'''
                x = float(splited[3 + 7 * i])
                y = float(splited[4 + 7 * i])
                x2 = float(splited[5 +  7* i])
                y2 = float(splited[6 + 7 * i])
                '''class'''
                c = splited[7 + 7 * i]

                if int(c) > - 1:
                    box.append([x, y, x2, y2])
                    label.append(int(c) + 1)

            if len(box) > 0:
                self.boxes.append(torch.Tensor(box))           # float 32
                self.labels.append(torch.LongTensor(label))    # int 64
                self.fnames.append(name)
        #
        self.num_samples = len(self.boxes)                 # total number of annotated bounding boxes
        self.indices = range(len(self.boxes))              # indices of the imagers which will be later used in getitem() 
                                                           # and mosaicing
    #########
    def __getitem__(self, idx):
        #
        '''this is the method for retrieving data with the iterator'''
        #
        index = self.indices[idx]
        #print(self.fnames[index])
        #print(index)
        
        if self.train:                                     # data augmentation only during training
            ''' all these are done in BGR'''
            
            '''image-depended mosaicing'''

            #boxes = self.boxes[index].clone()
            #mosaic_prob = 1.0 - self.image_weight_from_size(boxes)
            mosaic_prob = 0.7

            mosaic = random.random() < mosaic_prob

            if mosaic: 
                img, boxes, labels = self.load_mosaic_4(index)  # composite image

                '''
                if random.random() < 0.1:
                    img2, boxes2, labels2 = self.load_mosaic_4(random.randint(0, self.num_samples - 1))
                    r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                    img = (img * r + img2 * (1 - r)).astype(np.uint8)
                    labels = torch.cat((labels, labels2), dim = 0)
                    boxes = torch.cat((boxes, boxes2), dim = 0)

                '''

            else:
                img, _, boxes, labels = self.load_image(index)  # single image 

                '''
                if random.random() < 0.1:
                    img2, boxes2, labels2 = self.load_mosaic_4(random.randint(0, self.num_samples - 1))
                    r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                    img = (img * r + img2 * (1 - r)).astype(np.uint8)
                    labels = torch.cat((labels, labels2), dim = 0)
                    boxes = torch.cat((boxes, boxes2), dim = 0)
                '''
            
            img, boxes, labels = self.randomShift(img,boxes,labels)
            img, boxes, labels = self.randomZoom(img, boxes, labels, is_mosaic = mosaic)
            
            img, boxes = self.random_flip(img, boxes)
            img = self.randomBlur(img)

            img = self.hsv_augment(img)

            if random.random() < 0.5:
                img = self.hist_equalize(img)

        #

        else:   # test/val

            img, _, boxes, labels = self.load_image(index)  # single image with no augmentation
            
        '''resize to network dimension
        '''
        img, boxes = self.Resize(img, boxes)

        '''
        convert BGR to RGB, since pret-trained models use RGB
        '''
        
        img = self.BGR2RGB(img)
        
        '''
        fix b-box size if required
        '''
        boxes = boxes.clamp(0, self.image_size)            # keeping the box within the bound of the image
        
        '''important  !!!!!!!
           unusual boxes, such as, less than 2 pixels in either dimension,
           or unusually narrow or long, are excluded
        '''
        invalid = self.invalid_mask(boxes)
        boxes = boxes[invalid.expand_as(boxes)].view(-1, 4)
        labels = labels[invalid.view(-1)]

        
        '''
        weights = torch.zeros((boxes.shape[0], 1))
        for i, box in enumerate(boxes):

            area = (box[2] - box[0]) * (box[3] - box[1])
            
            if  area <= 32 ** 2:
                weights[i] = 3.0

            if  area > 32 ** 2 and area <= 96 ** 2:
                weights[i] = 2.0

            if  area > 96 ** 2:
                weights[i] = 1.0

        boxes = torch.cat([boxes, weights], dim = 1)
        '''

        ##### encoding


        #target, mask_obj, box_weight  = self.encoder(boxes, labels)
        target, mask_obj, mask_no_obj = self.encoder(boxes, labels)

        #print(boxes.shape)
        #print(labels.unsqueeze(dim = 1).shape)

        boxes = torch.cat((boxes, labels.float().unsqueeze(dim = 1) - 1.0), dim = 1)
        gt_boxes = torch.zeros((len(boxes), 6))
        gt_boxes[:, 1:] = boxes
        #
        if self.transform:
            img_n = self.transform(img)

        else:
            img_n = torch.tensor(img.copy())
        #
        #return img, boxes, labels, img_n, target
        return gt_boxes, img_n, target, mask_obj, mask_no_obj

    def collate_fn(self, batch):

        boxes, imgs, targets, mask_obj, mask_no_obj = list(zip(*batch))
        
        # Remove empty placeholder targets
        gt_boxes = [box for box in boxes if box is not None]
        # Add sample index to targets
        for i, box in enumerate(gt_boxes):
            box[:, 0] = i
        gt_boxes = torch.cat(gt_boxes, dim = 0)


        imgs = torch.stack(imgs)

        '''targets: a list of lists
           number of lists inside equals batch_size,
           each list has exactly two tensors (13 x 13 & 26 x 26)

        '''
        targets = list(zip(*targets))
        mask_obj = list(zip(*mask_obj))
        mask_no_obj = list(zip(*mask_no_obj))

        targets = [torch.stack(item) for item in targets]
        mask_obj = [torch.stack(item) for item in mask_obj]
        mask_no_obj = [torch.stack(item) for item in mask_no_obj]

        
        #return gt_boxes, imgs, targets, mask_obj, mask_no_obj
        return imgs, gt_boxes, targets, mask_obj, mask_no_obj

    ###########
    def __len__(self):
        return self.num_samples
    
    #
    def load_image(self, idx):

        fname = self.fnames[idx]
        '''read image (BGR) and b-boxes'''
        img = cv2.imread(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        #
        return img, img.shape[:2], boxes, labels
    #

    def image_weight_from_size(self, boxes):
    
        weights = torch.zeros((boxes.shape[0], 1))
        for i, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
                
            if  area <= 32 ** 2:
                weights[i] = 0.5

            if  area > 32 ** 2 and area <= 96 ** 2:
                weights[i] = 1.0 / 3.0

            if  area > 96 ** 2:
                weights[i] = 1.0 / 6.0

        return weights.mean().item()

    #

    def load_mosaic(self, idx):

        color = self.mean
        color = color[::-1] 
        
        boxes4, labels4 = [], []
        s = self.image_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [idx] + [self.indices[random.randint(0, self.num_samples - 1)] for _ in range(3)]  # 3 additional image indices

        
        for i, index in enumerate(indices):
            # Load image
            img, _, boxes, labels  = self.load_image(index)
            #img, boxes = self.resizeMosaic(img, boxes)
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), color, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            #
            #mask = self.get_mask(boxes, x1b, y1b, x2b, y2b)
            #boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
            #labels = labels[mask.view(-1)]
            #
            if boxes.size:  ## if all boxes are not expelled
                boxes = self.xywhn2xyxy(boxes, padw, padh) 

                boxes4.append(boxes)
                labels4.append(labels)

        boxes4 = torch.cat(boxes4, 0)
        labels4 = torch.cat(labels4, 0)

        #boxes4 = boxes4.clamp(0, 2 * s)

        
        #
        return img4, boxes4, labels4

    def load_mosaic_2(self, idx):

        color = self.mean
        color = color[::-1] 
        
        boxes2, labels2 = [], []
        s = self.image_size
        yc = 0
        xc = int(random.uniform(0, 640))  # mosaic center x, y
        #print(xc)
        indices = [idx] + [self.indices[random.randint(0, self.num_samples - 1)] for _ in range(1)]  # 3 additional image indices

        
        for i, index in enumerate(indices):
            # Load image
            img, _, boxes, labels  = self.load_image(index)
            #print(boxes.shape)
            #img, boxes = self.resizeMosaic(img, boxes)
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img2 = np.full((h, w * 2, img.shape[2]), color, dtype=np.uint8)  # base image with 2 tiles

                img2[:, 0 : w] = img

                plt.imshow(img2)
                plt.show()

                #print(boxes)

                boxes[:, 0] = torch.min(torch.max(torch.tensor(0), boxes[:, 0] + 0 - xc), torch.tensor(w - 1)) # top left x
                boxes[:, 1] = torch.min(torch.max(torch.tensor(0), boxes[:, 1] + 0 - yc), torch.tensor(h - 1))  # top left y
                boxes[:, 2] = torch.min(torch.max(torch.tensor(0), boxes[:, 2] + 0 - xc), torch.tensor(w - 1)) # bottom right x
                boxes[:, 3] = torch.min(torch.max(torch.tensor(0), boxes[:, 3] + 0 - yc), torch.tensor(h - 1))  # bottom right y

                mask = self.get_mask(boxes, 0, 0, w, h)
                boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
                labels = labels[mask.view(-1)]

            elif i == 1:  # top right
                img2[:, w : ] = img

                plt.imshow(img2)
                plt.show()

                boxes[:, 0] = torch.min(torch.max(torch.tensor(0), boxes[:, 0] + w - xc), torch.tensor(w - 1)) # top left x
                boxes[:, 1] = torch.min(torch.max(torch.tensor(0), boxes[:, 1] + 0 - yc), torch.tensor(h - 1))  # top left y
                boxes[:, 2] = torch.min(torch.max(torch.tensor(0), boxes[:, 2] + w - xc), torch.tensor(w - 1))  # bottom right x
                boxes[:, 3] = torch.min(torch.max(torch.tensor(0), boxes[:, 3] + 0 - yc), torch.tensor(h - 1))  # bottom right y


                print(boxes)

                mask = self.get_mask(boxes, 0, 0, w, h)
                boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
                labels = labels[mask.view(-1)]

                print(boxes)

            #print(img2.shape)
            #print(boxes)

            
            #print(boxes)
            
            if boxes.size:  ## if all boxes are not expelled
                #boxes = self.xywhn2xyxy(boxes, padw, padh) 

                
                boxes2.append(boxes)
                labels2.append(labels)

        img2 = img2[:, xc:xc+w]
        plt.imshow(img2)
        plt.show()

        boxes2 = torch.cat(boxes2, 0)
        labels2 = torch.cat(labels2, 0)

        #boxes4 = boxes4.clamp(0, 2 * s)

        
        #
        return img2, boxes2, labels2

    def load_mosaic_4(self, idx):

        color = self.mean
        color = color[::-1] 
        
        boxes4, labels4 = [], []
        
        
        #print(xc)
        indices = [idx] + [self.indices[random.randint(0, self.num_samples - 1)] for _ in range(3)]  # 3 additional image indices

        for i, index in enumerate(indices):

            img, (h, w), boxes, labels  = self.load_image(index)

            if i == 0:  # top left

                h_, w_ = h, w
                
                img4 = np.full((h_ * 2, w_ * 2, img.shape[2]), color, dtype=np.uint8)  # base image with 2 tiles


                yc = int(random.uniform(0, h_))
                xc = int(random.uniform(0, w_))  # mosaic center x, y

                img4[: h, : w] = img

                boxes[:, 0] = boxes[:, 0] + 0 - xc
                boxes[:, 1] = boxes[:, 1] + 0 - yc 
                boxes[:, 2] = boxes[:, 2] + 0 - xc
                boxes[:, 3] = boxes[:, 3] + 0 - yc


            elif i == 1:  # top right

                img4[h_ - min(h_, h): h_, w_ : w_ + min(w_, w)] = img[0:min(h_, h), 0: min(w_, w)]

                boxes[:, 0] = boxes[:, 0] + w_ - xc
                boxes[:, 1] = boxes[:, 1] + h_ - min(h_, h) - yc 
                boxes[:, 2] = boxes[:, 2] + w_ - xc
                boxes[:, 3] = boxes[:, 3] + h_ - min(h_, h) - yc


            elif i == 2:  # bottom left

                img4[h_ : h_ + min(h_, h), w_ -  min(w_, w): w_] = img[0:min(h_, h), 0: min(w_, w)]

                boxes[:, 0] = boxes[:, 0] + w_ -  min(w_, w) - xc
                boxes[:, 1] = boxes[:, 1] + h_ - yc
                boxes[:, 2] = boxes[:, 2] + w_ -  min(w_, w) - xc
                boxes[:, 3] = boxes[:, 3] + h_ - yc


            elif i == 3:  # bottom right
                
                img4[h_ : h_ + min(h_, h), w_ : w_ + min(w_, w)] = img[0:min(h_, h), 0: min(w_, w)]

                boxes[:, 0] = boxes[:, 0] + w_ - xc
                boxes[:, 1] = boxes[:, 1] + h_ - yc
                boxes[:, 2] = boxes[:, 2] + w_ - xc
                boxes[:, 3] = boxes[:, 3] + h_ - yc

            
            boxes[:, 0] = torch.min(torch.max(torch.tensor(0), boxes[:, 0]), torch.tensor(w_ - 1)) 
            boxes[:, 1] = torch.min(torch.max(torch.tensor(0), boxes[:, 1]), torch.tensor(h_ - 1))  
            boxes[:, 2] = torch.min(torch.max(torch.tensor(0), boxes[:, 2]), torch.tensor(w_ - 1))  
            boxes[:, 3] = torch.min(torch.max(torch.tensor(0), boxes[:, 3]), torch.tensor(h_ - 1))   

            mask = self.get_mask(boxes, 0, 0, w_, h_)
            boxes = boxes[mask.expand_as(boxes)].view(-1, 4)
            labels = labels[mask.view(-1)]

            
            
            if boxes.size:  ## if all boxes are not expelled
                
                boxes4.append(boxes)
                labels4.append(labels)

        #plt.imshow(img4)
        #plt.show()

        img4 = img4[yc : yc + h_:, xc : xc + w_]
        #plt.imshow(img4)
        #plt.show()

        boxes4 = torch.cat(boxes4, 0)
        labels4 = torch.cat(labels4, 0)

        #
        return img4, boxes4, labels4

    def xywhn2xyxy(self, x, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] + padw  # top left x
        y[:, 1] = x[:, 1] + padh  # top left y
        y[:, 2] = x[:, 2] + padw  # bottom right x
        y[:, 3] = x[:, 3] + padh  # bottom right y
        
        return y

    def invalid_mask(self, boxes, wh_th = 2, area_th = 200, ar_th = 20):

        eps = 1e-9 

        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]      # width/height

        area = w * h                                                  # area
    
        ar = torch.max(w / (h + eps), h / (w + eps))              # aspect ratio

        mask = (w > wh_th) & (h > wh_th) & (area > area_th) & (ar < ar_th)

        return mask.view(-1, 1)

    def get_mask(self, boxes, x1, y1, x2, y2):

        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        
        x, y, w, h = x1, y1, x2 - x1, y2 - y1        ## correction made june 19
        #
        center = center - torch.FloatTensor([[x, y]]).expand_as(center)
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)

        mask = (mask1 & mask2).view(-1, 1)
        #
        return mask

    #################################################################################################################
    #################################################################################################################
    '''helper functions'''
    def bbox_wh_iou(self, wh1, wh2):
        
        #wh2 = wh2.t()
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1 + self.epsilon) + w2 * h2 - inter_area

        return inter_area / union_area


    def _encoder(self, boxes, labels, anchors, grid_num):

        stride = self.image_size // grid_num
        anchors = torch.tensor(anchors) / stride           # normalized anchors

        num_anchors = len(anchors)
        bbox_attrs = 5 + self.num_classes

        '''target 3 dimensional'''
        target = torch.FloatTensor(num_anchors, grid_num, grid_num, bbox_attrs).fill_(0)
        
        object_mask = torch.BoolTensor(num_anchors, grid_num, grid_num, bbox_attrs).fill_(False)       # should be expanded to all channels and anchor dimensions
        no_object_mask = torch.BoolTensor(num_anchors, grid_num, grid_num, bbox_attrs).fill_(True) 
        
        #
        #print(boxes)
        '''find the center coordinate of boxes and their width and height

           and also normalize by network stride

        '''
        '''
        variable boxes should be a tensor
        '''
        wh = boxes[:, 2:] / stride - boxes[:, :2] / stride
        cxcy = (boxes[:, 2:] / stride + boxes[:, :2] / stride) / 2
        #
        '''the center of the bounding boxes will be encoded as the offset from the nearest grid cell location
           the width and height will be encoded as factors of the width abnd height of the nearest anchor box (with highest IoU) 

        '''
        for i in range(cxcy.size()[0]):

            '''instead of IoU use the ratio of width or height'''

            '''from ultralytics yolov5 Glenn Jocher'''
            r = torch.stack([wh[i] / anchor for anchor in anchors])
            r = torch.max(r, 1. / r).max(dim = 1)[0] < self.anchor_t

            cxcy_sample = cxcy[i]
            ij = cxcy_sample.floor() #

            
            object_mask[r, int(ij[1]), int(ij[0]), :] = True 
            no_object_mask[r, int(ij[1]), int(ij[0]), :] = False

            target[r, int(ij[1]), int(ij[0]), 4] = 1
            target[r, int(ij[1]), int(ij[0]), int(labels[i]) + 4] = 1
                    
            target[r, int(ij[1]), int(ij[0]), 2:4] = torch.log(wh[i] / anchors[r] + self.epsilon)
            target[r, int(ij[1]), int(ij[0]), :2] = (cxcy_sample - ij) / self.alpha + (self.alpha - 1.0) / (2 * self.alpha)

            '''    
            object_mask[best_n, int(ij[1]), int(ij[0]), :] = True 
            no_object_mask[best_n, int(ij[1]), int(ij[0]), :] = False
            
            target[best_n, int(ij[1]), int(ij[0]), 4] = 1
            target[best_n, int(ij[1]), int(ij[0]), int(labels[i]) + 4] = 1
            
            target[best_n, int(ij[1]), int(ij[0]), 2:4] = torch.log(wh[i] / anchors[best_n] + self.epsilon)
            target[best_n, int(ij[1]), int(ij[0]), :2] = (cxcy_sample - ij) / self.alpha + (self.alpha - 1.0) / (2 * self.alpha)
            '''

            
            #
        
        
        return target, object_mask, no_object_mask



    def encoder(self, boxes, labels):
        
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        
        '''

        # anchor is either a list of two lists or a list of tuples
        
        # when list of lists

        self.image_size // self.stride

        if len(self.anchors) == 3:
            grid_num = [self.image_size // self.stride, 
                        2 * self.image_size // self.stride, 
                        4 * self.image_size // self.stride
                       ]
        
        elif len(self.anchors) == 2:
            grid_num = [self.image_size // self.stride, 
                        2 * self.image_size // self.stride, 
                       ]
        
        # otherwise
        else:
            grid_num = self.image_size // self.stride

        
        if not isinstance(grid_num, list):

            target, mask_obj, mask_no_obj = self._encoder(boxes, labels, anchors = self.anchors, grid_num = grid_num)                                      # 7x7x30


        else:

            target, mask_obj, mask_no_obj = [], [], []      # list of tensor objects

            for anchors, grid in zip(self.anchors, grid_num):

                target_, mask_obj_, mask_no_obj_ = self._encoder(boxes, labels, anchors = anchors, grid_num = grid)

                target.append(target_)
                mask_obj.append(mask_obj_)
                mask_no_obj.append(mask_no_obj_)
        
        return target, mask_obj, mask_no_obj
    
    #
    def BGR2RGB(self,img):
        '''cv2 reads BGR - convert to RGB'''
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    def BGR2HSV(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    def HSV2BGR(self,img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    #
    def BGR2YUV(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #
    def YUV2BGR(self,img):
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    #
    def hsv_augment(self, bgr):

        dtype = bgr.dtype  # uint8
        hsv = self.BGR2HSV(bgr)
        hue, sat, val = cv2.split(hsv)

        #https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        #r = np.random.uniform(-1, 1, 3) * [0.5, 0.5, 0.5] + 1  # random gains
        r = np.random.uniform(-1, 1, 3) * [0.02, 0.7, 0.4] + 1  # random gains
        
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        bgr = self.HSV2BGR(hsv)

        #
        return bgr
    
    #
    def hist_equalize(self, bgr, clahe = True):
        #https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
        yuv = self.BGR2YUV(bgr)
        
        if clahe:  # contrast-limited adaptive histogram equalization
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
        #
        bgr = self.YUV2BGR(yuv)
        
        return bgr

    #
    def randomBlur(self,bgr):
        #
        if random.random() < 0.5:
            k = 2 * random.randint(1, 7) + 1
            bgr = cv2.blur(bgr,(k, k))
        #
        return bgr
    #
    def randomShift(self, bgr, boxes, labels):
        
        bgr_in = bgr.copy()
        boxes_in = boxes.detach().clone()
        labels_in = labels.detach().clone()

        '''image diemsnions'''
        height, width, _ = bgr.shape
        
        ''' translation matrix'''
        shift_x = random.uniform(-width * 0.2, width * 0.2)
        shift_y = random.uniform(-height * 0.2, height * 0.2)
        #
        M = np.eye(2, 3)
        M[0, 2] = shift_x
        M[1, 2] = shift_y

        '''apply translation'''
        border_value = self.mean
        border_value = border_value[::-1] 

        bgr = cv2.warpAffine(bgr, M, (width, height), borderValue = border_value)    # keep the dimension of the original
                                                                                     # assign boder values to data set mean

        '''now translate the boxes and labels
           exclude boxes and labels that go out of bound
        '''

        '''first, convert boxes to numpy'''
        boxes = boxes.reshape((-1, 2))                                                    # N x 4 -> 2N x 2
        boxes = boxes.detach().numpy()
        boxes = np.hstack([boxes, np.ones((len(boxes), 1))])                         # concate 1s to (x, y) coordinates

        # transform
        boxes = M.dot(boxes.T).T                                                      
        boxes = np.clip(boxes, np.array([0., width]), np.array([0., height]))        # clip out of bound boxes
                                                                                     # some boxes will completely go out  
                                                                                     # need to delete them & their corresponding labels
        '''back to torch tensor;
           and apply a mask to exclude invalid boxes & labels
        '''
        boxes = torch.from_numpy(boxes)                                              # back to torch
        boxes = boxes.reshape((-1, 4))                                                   # 2N x 2 -> N x 4

        '''box centers '''
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

        mask1 = (cxcy[:, 0] > 0) & (cxcy[:, 0] < width)
        mask2 = (cxcy[:, 1] > 0) & (cxcy[:, 1] < height)
        mask = (mask1 & mask2).view(-1,1)                                            # mask to retain

        boxes = boxes[mask.expand_as(boxes)].view(-1,4)                              # retained boxes
        labels = labels[mask.view(-1)]                                               # retained labels

        if len(boxes) == 0:
            '''if the shift ends up losing all the b-boxes,
                   discard it and return the unchanged image and labels
            '''
            return bgr_in, boxes_in, labels_in
        
        else:

            return bgr, boxes, labels

    #
    def randomZoom(self, bgr, boxes, labels, is_mosaic = False):
        #

        bgr_in = bgr.copy()
        boxes_in = boxes.detach().clone()
        labels_in = labels.detach().clone()

        '''image diemsnions'''
        height, width, _ = bgr.shape

        '''transformation matrix'''
        if is_mosaic:
            scale = random.uniform(0.3, 1.5)
            bgr_ctr = (int(width / 2 + random.uniform(-200, 200)), int(height / 2 + random.uniform(-200, 200)))

        else:
            scale = random.uniform(0.1, 1.5)
            bgr_ctr = (int(width / 2 + random.uniform(-50, 50)), int(height / 2 + random.uniform(-50, 50)))

        M = cv2.getRotationMatrix2D(angle = 0, center = bgr_ctr, scale = scale)

        '''apply transformation'''
        border_value = self.mean
        border_value = border_value[::-1] 

        bgr = cv2.warpAffine(bgr, M, (width, height), borderValue = border_value)    # keep the dimension of the original
                                                                                     # assign boder values to data set mean

        '''now translate the boxes and labels
           exclude boxes and labels that go out of bound
        '''

        '''first, convert boxes to numpy'''
        boxes = boxes.reshape((-1, 2))                                                  # N x 4 -> 2N x 2
        boxes = boxes.detach().numpy()
        boxes = np.hstack([boxes, np.ones((len(boxes), 1))])                         # concate 1s to (x, y) coordinates

        # transform
        boxes = M.dot(boxes.T).T                                                      
        boxes = np.clip(boxes, np.array([0., width]), np.array([0., height]))        # clip out of bound boxes
                                                                                     # some boxes will completely go out  
                                                                                     # need to delete them & their corresponding labels
        '''back to torch tensor;
           and apply a mask to exclude invalid boxes & labels
        '''
        boxes = torch.from_numpy(boxes)                                              # back to torch
        boxes = boxes.reshape((-1, 4))                                                    # 2N x 2 -> N x 4

        '''box centers '''
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

        mask1 = (cxcy[:, 0] > 0) & (cxcy[:, 0] < width)
        mask2 = (cxcy[:, 1] > 0) & (cxcy[:, 1] < height)
        mask = (mask1 & mask2).view(-1,1)                                            # mask to retain

        boxes = boxes[mask.expand_as(boxes)].view(-1,4)                              # retained boxes
        labels = labels[mask.view(-1)]                                               # retained labels

        if len(boxes) == 0:
            '''if the shift ends up losing all the b-boxes,
                   discard it and return the unchanged image and labels
            '''
            return bgr_in, boxes_in, labels_in
        
        else:

            return bgr, boxes, labels
    

    def resizeMosaic(self, rgb, boxes):
        #
        '''find the ratio'''
        old_size = rgb.shape[:2]
        desired_size = self.image_size

        ratio = 1
        #
        if not max(old_size) == self.image_size:
  
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            #
            '''now resize the image'''
            rgb = cv2.resize(rgb, (new_size[1], new_size[0]))                               # remember: resize takes (W, H)
            #
        #
        #
        '''do not forget to resize the b-boxes'''  
        #print(ratio)                                  
        scale_tensor = torch.FloatTensor([[ratio, ratio, ratio, ratio]]).expand_as(boxes)
        boxes = boxes * scale_tensor
        #
        return rgb, boxes    

    #
    '''method for image resize to self.image_size
       https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    '''
    def Resize(self, rgb, boxes):
        #
        '''find the ratio'''
        old_size = rgb.shape[:2]
        desired_size = self.image_size
        #
        if max(old_size) < self.image_size:
            delta_w = desired_size - old_size[1]
            delta_h = desired_size - old_size[0]
            ratio = 1.0
        else:
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            #
            '''now resize the image'''
            rgb = cv2.resize(rgb, (new_size[1], new_size[0]))                               # remember: resize takes (W, H)
            #
            '''now find the padding size '''
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
        #

        if self.train:
            a, b = random.uniform(0, 1.0),  random.uniform(0, 1.0)
            top = int(a * delta_h)
            bottom = delta_h - top
            left = int(b * delta_w)
            right = delta_w - left
        else:
            top, bottom = 0, delta_h
            left, right = 0, delta_w
        
        #top, bottom = 0, delta_h
        #left, right = 0, delta_w
        
        #
        '''set the padding value to data mean'''
        color = self.mean
        color = color[::-1] 
        #
        '''now, do the padding'''
        rgb_pad = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)
        #
        '''do not forget to resize the b-boxes'''  
        #print(ratio)
        box_shift = torch.FloatTensor([[left, top, left, top]]).expand_as(boxes)                                     
        scale_tensor = torch.FloatTensor([[ratio, ratio, ratio, ratio]]).expand_as(boxes)
        boxes = boxes * scale_tensor + box_shift
        #boxes = boxes * scale_tensor
        #
        return rgb_pad, boxes
    #
    def random_flip(self, im, boxes):
        #
        '''horizontal flip'''
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            #
            im = im_lr
        #
        return im, boxes