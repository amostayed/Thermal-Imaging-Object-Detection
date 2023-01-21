import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import models

import numpy as np
import math

from src.networks.modules import *



class Neck(nn.Module):

    def __init__(self):
        '''yolov5s backbone as defined in:
           https://github.com/ultralytics/yolov5/blob/master/models/yolov5m.yaml
        '''
        #
        super(Neck, self).__init__()
        #

        self.rout_to = [1, 5, 8, 11]

        '''define the network layers'''
        neck = []                      # initialize
        
        '''upsample & concat'''
        neck += [nn.Upsample(scale_factor = 2.0, mode = 'nearest'),
                   Concat()]      # p/16
        
        '''C3 block: 1 bottlenek'''
        c1, c2, n, e = (768, 384, 2, 0.5)      # n-> number of bottlenecks; e-> expansion rate
        neck.append(C3(c1, c2, n, e = e))
        
        '''standard conv-bn-act: 512 -> 256, stride = 1, k = 1'''
        c1, c2, k, s = (384, 192, 1, 1)
        neck.append(Conv(c1, c2, k, s))
        
        '''upsample & concat'''
        neck += [nn.Upsample(scale_factor = 2.0, mode = 'nearest'),
                   Concat()]      # p/8
        
        '''C3 block: 1 bottlenek'''
        c1, c2, n, e = (384, 192, 2, 0.5)      # n-> number of bottlenecks; e-> expansion rate
        neck.append(C3(c1, c2, n, e = e))
        
        '''standard conv-bn-act: 128 -> 128, stride = 2, k = 3'''
        c1, c2, k, s = (192, 192, 3, 2)
        neck.append(Conv(c1, c2, k, s))
        
        '''concat'''
        neck.append(Concat())    # p/16
        
        '''C3 block: 1 bottlenek'''
        c1, c2, n, e = (384, 384, 2, 0.5)      # n-> number of bottlenecks; e-> expansion rate
        neck.append(C3(c1, c2, n, e = e))
        
        '''standard conv-bn-act: 256 -> 256, stride = 2, k = 3'''
        c1, c2, k, s = (384, 384, 3, 2)
        neck.append(Conv(c1, c2, k, s))
        
        '''concat'''
        neck.append(Concat())     # p/32
        
        '''C3 block: 1 bottlenek'''
        c1, c2, n, e = (768, 768, 1, 0.5)      # n-> number of bottlenecks; e-> expansion rate
        neck.append(C3(c1, c2, n, e = e))

        self.m = nn.Sequential(*neck)


    def forward(self, X):
        #
        x = X[-1]

        layers = []
        
        for i, m in enumerate(self.m):

            if i == 1:
                x = [x, X[1]]

            if i == 5:
                x = [x, X[0]]

            if i == 8:
                x = [x, X[2]]

            if i == 11:
                x = [x, X[-1]]

            x = m(x)

            if i == 3:
                X.insert(2, x)

            if i in [6, 9, 12]:
                layers.append(x)
            
        
        return layers

class Backbone(nn.Module):

    def __init__(self):
        '''yolov5s backbone as defined in:
           https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml
        '''
        #
        super(Backbone, self).__init__()
        #

        self.rout_from = [6, 4, 10]

        '''define the network layers'''
        backbone = []                      # initialize
        
        '''Focus: converts 3-channel input to 12-channel input & 
           reducing spatial dimensions by half in the process (leverages the high spatial correlation between pixels);
           then applies a standard conv-bn-act; SilU activation
        '''
        c1, c2, k, s = (3, 48, 3, 1)
        backbone.append(Focus(c1, c2, k, s))       # p/2
        
        '''standard conv-bn-act: 32 -> 64, stride = 2, k = 3'''
        c1, c2, k, s = (48, 96, 3, 2)
        backbone.append(Conv(c1, c2, k, s))        # p/4
        
        '''C3 block: CSP bottleneck with 3 convolutions'''
        c1, c2, n, e = (96, 96, 2, 0.5)      # n-> number of bottlenecks; e-> expansion rate
        backbone.append(C3(c1, c2, n, e = e))
        
        '''standard conv-bn-act: 64 -> 128, stride = 2, k = 3'''
        c1, c2, k, s = (96, 192, 3, 2)             # p/8
        backbone.append(Conv(c1, c2, k, s))
        
        '''C3 block: 3 bottleneks'''
        c1, c2, n, e = (192, 192, 6, 0.5)      # n-> number of bottlenecks; e-> expansion rate
        backbone.append(C3(c1, c2, n, e = e))
        
        '''standard conv-bn-act: 128 -> 256, stride = 2, k = 3'''
        c1, c2, k, s = (192, 384, 3, 2)
        backbone.append(Conv(c1, c2, k, s))        # p/16
        
        '''C3 block: 3 bottleneks'''
        c1, c2, n, e = (384, 384, 6, 0.5)      # n-> number of bottlenecks; e-> expansion rate
        backbone.append(C3(c1, c2, n, e = e))
        
        '''standard conv-bn-act: 256 -> 512, stride = 2, k = 3'''
        c1, c2, k, s = (384, 768, 3, 2)
        backbone.append(Conv(c1, c2, k, s))        # p/32
        
        '''SPP: pyramid pooling '''
        c1, c2, k = (768, 768, (3, 7, 9))
        backbone.append(SPP(c1, c2, k))
        
        '''C3 block: 3 bottleneks'''
        c1, c2, n, e = (768, 768, 2, 0.5)      # n-> number of bottlenecks; e-> expansion rate
        backbone.append(C3(c1, c2, n, e = e))
        
        '''standard conv-bn-act: 512 -> 256, stride = 1, k = 1'''
        c1, c2, k, s = (768, 384, 1, 1)
        backbone.append(Conv(c1, c2, k, s))       # p/32


        self.m = nn.Sequential(*backbone)

    def forward(self, x):
        #
        layers = []
        
        for i, m in enumerate(self.m):

            x = m(x)
            
            if i in self.rout_from:
                layers.append(x)

        return layers   #[p/8, p/16, p/32]

#


class Head(nn.Module):

    def __init__(self, in_ch, anchor, cls_probs, num_classes = 20, inp_dim = 416):
        #
        super(Head, self).__init__()
    
        
        #
        self.in_ch = in_ch
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.num_anchors = [len(a) for a in anchor][0]
        #self.num_anchors = len(anchor)
        self.cf = cls_probs 
        #

        # classification
        conv_cls = nn.Conv2d(self.in_ch, self.num_anchors * self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        b = conv_cls.bias.view(self.num_anchors, -1)  
        b.data += math.log(0.6 / (self.num_classes - 0.99)) if self.cf is None else torch.log(self.cf / self.cf.sum()) 
        conv_cls.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        # objectness head
        conv_obj = nn.Conv2d(self.in_ch, self.num_anchors, kernel_size=1, stride=1, padding=0, bias=True)
        #b = conv_obj.bias.view(self.num_anchors, -1)  
        #b.data += math.log(8 / (inp_dim / 8) ** 2)
        #conv_obj.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        
        # regression head
        conv_box = nn.Conv2d(self.in_ch, self.num_anchors * 4, kernel_size=1, stride=1, padding=0, bias=True)
        
        # localization head
        conv_loc = nn.Conv2d(self.in_ch, self.num_anchors, kernel_size=1, stride=1, padding=0, bias=True)
        b = conv_loc.bias.view(self.num_anchors, -1)  
        b.data += -5.0
        conv_loc.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        
        self.m = nn.ModuleList(
                                  [nn.Sequential(
                                                 #Conv(self.in_ch, self.in_ch // 2, 1, 1), 
                                                 #Conv(self.in_ch // 2, self.in_ch, 3, 1),
                                                 #C3(self.in_ch, self.in_ch, n = 1, e = 1.0),
                                                 conv_box
                                                ), 
                                  nn.Sequential(
                                                 #Conv(self.in_ch, self.in_ch // 8, 1, 1), 
                                                 #Conv(self.in_ch // 8, self.in_ch, 3, 1),
                                                 #C3(self.in_ch, self.in_ch, n = 1, e = 0.25),
                                                 conv_obj
                                                ), 
                                  nn.Sequential(
                                                 #Conv(self.in_ch, self.in_ch // 8, 1, 1), 
                                                 #Conv(self.in_ch // 8, self.in_ch, 3, 1),
                                                 #C3(self.in_ch, self.in_ch, n = 1, e = 0.5),
                                                 conv_loc
                                                ),
                                  nn.Sequential(
                                                 #Conv(self.in_ch, self.in_ch // 4, 1, 1), 
                                                 #Conv(self.in_ch // 4, self.in_ch, 3, 1),
                                                 #C3(self.in_ch, self.in_ch, n = 1, e = 0.25),
                                                 conv_cls
                                                )
                                  ]
                              )

    def forward(self, x):
        #
        
        out = torch.cat([m(x) for m in self.m], 1)

        
        return out


class Output(nn.Module):
    
    def __init__(self, in_ch, anchor, cls_probs, num_classes = 20, inp_dim = 416):
        #
        super(Output, self).__init__()
    
        
        #
        self.in_ch = in_ch
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.cf = cls_probs 
        #
        self.head = Head(self.in_ch, anchor, self.cf, self.num_classes, self.inp_dim)

        self.s3 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.s4 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.s5 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.s = [self.s5, self.s4, self.s3]


    def forward(self, X):
        #

        outputs = len(X) * [None]
        
        #
        for i, x in enumerate(X):
            
            out = self.head(x)

            out[:, 2:4, :, :] = self.s[i] * out[:, 2:4, :, :].clone()

            batch, dim = out.size(dim = 0), out.size(dim = 2)
            out = out.view(batch, -1, self.num_classes + 6, dim, dim)
            out = out.permute(0, 1, 3, 4, 2)

            outputs[-(i+1)] = out
        
        return outputs


class Yolo(nn.Module):
    '''yolov5s backbone as defined in:
       https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml
    '''
    def __init__(self, anchors, cls_probs, num_classes = 20, inp_dim = 416):
        #
        super(Yolo, self).__init__()
        
        #self.out = [17, 20, 23]
        #self.rout_from = [6, 4, 14, 10]
        #self.rout_to = [12, 16, 19, 22]
        
        #
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = [len(anchors) for anchors in self.anchors]
        self.cf = cls_probs 
        
        self.backbone = Backbone()

        self.neck = Neck()

        reduction_ch = 256
        self.reduction = nn.ModuleList([Conv(192, reduction_ch, 1, 1), 
                                        Conv(384, reduction_ch, 1, 1),  
                                        Conv(768, reduction_ch, 1, 1),]
                                      )

        self.head = Output(reduction_ch, self.anchors[::-1], self.cf, self.num_classes, self.inp_dim)
        
    def forward(self, x):
        #

        feature_pyramid = self.neck(self.backbone(x))

        return self.head([module(feature_pyramid[i]) for i, module in enumerate(self.reduction)])


    def load_pretrained_weight(self, pretrained_path):

        # model : nn.Module
        # pretrained_path : path of a '.pth' file # (e.g. yolov5s.pth)
        
        # assumption: pre-trained model (separate pyramid heads) has more layers than the current model (shared head)
        # otherwise, weights will not be loaded correctly
        
        
        pre_trained_dict = torch.load(pretrained_path)
        
        model_dict = self.state_dict()
        keys = list(model_dict.items())

        
        count = 0
        success = 0

        for k, v in pre_trained_dict.items():

            if not count > len(keys) - 1:

                key = keys[count][0]

                #print("pretrained:", k)

                #print("current:", key)

                
                
                _, weights = keys[count]

                if weights.shape == v.shape:
                    #print("matched:",  key)
                    
                    model_dict[key] = v

                    success += 1

                else:
                    pass
                    #print("did not matched:",  key)
                
                #print("################################")

                count += 1

        
        self.load_state_dict(model_dict)
        
        print('Total model keys: {}'.format(len(list(model_dict.items()))))
        print('Sucessfully matched: {}'.format(success))