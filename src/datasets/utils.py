''' generate weights for weighted sampler '''



import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from itertools import chain


def image_weights_from_label_file(file, classes_weights = None, num_classes = 20):

    labels = []

    with open(file) as f:        

        lines  = f.readlines()
        #
        for line in lines:
            if line.startswith('#'):
                continue
            splited = line.strip().split()

            if not splited or len(splited) < 8:
                continue
            
            num_boxes = (len(splited) - 1) // 7
            
            label = []
            
            for i in range(num_boxes):
                
                '''class'''
                c = splited[7 + 7 * i]
                
                
                if int(c) > - 1:
                    label.append(int(c))

            if len(label) > 0:
                labels.append(label)
                
    if classes_weights is None:       
        classes = list(chain(*labels))
        classes = np.array(classes).astype(np.int)  
                
        classes_weights = np.bincount(classes, minlength = num_classes)
        
        classes_weights = 1 / classes_weights  
        classes_weights /= classes_weights.sum()  # normalize


            
    classes_counts = np.array([np.bincount(np.array(x).astype(np.int), minlength = num_classes) for x in labels])
    object_counts = classes_counts.sum(axis = 1)
    object_counts = np.where(object_counts > 0, object_counts, 1)  # fix for image with no annotated object
                                                                   # such images will be excluded from sampler
            
    image_weights = (classes_weights.reshape(1, num_classes) * classes_counts).sum(axis = 1) / object_counts
    #image_weights = np.delete(image_weights, image_weights.argmin())
    #image_weights /= image_weights.sum()     # normalize
    
    #image_weights = np.clip(image_weights, 5e-5, 1e-4)
            
            
    return image_weights 