import os
import json
from os import listdir, getcwd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re



'''
This script converts the JSON label files (index.json)

'''



categoryFrequency = {'person': 50748, 
                     'bike': 7237, 
                     'car': 73623, 
                     'motor': 1116, 
                     'bus': 2245, 
                     'truck': 829, 
                     'light': 16198, 
                     'hydrant': 1095, 
                     'sign': 20770, 
                     'other_vehicle': 1373}


category2class = {'person': 0, 
                  'bike': 1, 
                  'car': 2, 
                  'motor': 3, 
                  'bus': 4, 
                  'truck': 5, 
                  'light': 6, 
                  'hydrant': 7, 
                  'sign': 8, 
                  'other_vehicle': 9}


def getFrame(frame):

  '''
  returns farme name and annotations
  annotations is a list of objects
  with attributes
  '''
    
  # frame meta data
  frame_meta = frame['videoMetadata']
  videoId = frame_meta['videoId']
  frameIndex = "{:06d}".format(frame_meta['frameIndex'])

  FrameId = frame['datasetFrameId']
  frame_annotations = frame['annotations']
    
  frame_name = 'video-' + videoId + '-frame-' + frameIndex + '-' + FrameId + '.jpg'
    
  return frame_name, frame_annotations


# parse the truncation field
def getTruncation(strn):
    
  '''% truncation'''
    
  return strn.split('%')[0]

# parse the occlusion field
def getOcclusion(strn):
    
  return '0' if ('fully' in strn or 'partially' in strn) else '1'


# parse the object bounding box
def getBox(box_dict):

  '''
  bounding box is stored in a dictionary
  returns list in x y x1 y1 format

  '''
    
  top_left = [box_dict['x'], box_dict['y']]
  bottom_right = [box_dict['x'] + box_dict['w'], box_dict['y'] + box_dict['h']]
    
  box = [str(b) for b in top_left + bottom_right]
    
  return ' '.join(box)


# parse the object category
def getLabel(label_list):

  '''
  label is a list

  '''

  label_list = label_list[0].split(' ')

  if len(label_list) > 1:
    label = '_'.join(label_list)

  else:
    label = label_list[0]

  if label in category2class:
    label = str(category2class[label])
  else:
    # do not parse the infrequent categories
    # at the moment hard-coded through category2class
    # TODO: set a frequncy threshold
    label = str(-1)

  return label



# converts the JSON entry for an object to a string 
def getAnnotation(annotation):
    
  ''' 
  Each object will be stored in the following format :
  occlusion | truncation | x | y | x2 | y2 | category
        
  amount of occlusion is  => 'occluded': (fully_visible) -> 0 | (partially_occluded) -> 0 | (difficult_to_see)  -> 1
  truncation => 'truncated'
    
  '''
  #
  ''' the 'custom' field indicates occlusion & truncation'''
  if 'custom' in annotation:
    occlusion_or_truncation = annotation['custom']
    if 'occluded' in occlusion_or_truncation:
      occlusion = getOcclusion(occlusion_or_truncation['occluded'])
    else:
      occlusion = '0'

    if 'truncated' in occlusion_or_truncation:
      truncation = getTruncation(occlusion_or_truncation['truncated'])
    else:
      truncation = '0'
            
  else:
    occlusion = '0'
    truncation = '0'
            
  # category
  label = getLabel(annotation['labels'])
            
  # bounding box
  box = getBox(annotation['boundingBox'])
  
  return ' '.join([occlusion, truncation, box, label])



# converts the JSON (index.json) to a text file
# each line of the text is one frame
def json2txt(json_in_file, txt_out_file):
  with open(json_in_file,'r') as f:
    parse = json.load(f)
        
    
  with open(txt_out_file, 'w') as f:
    #frames is a list holding all the frames
    for frame in parse['frames']:
      # retrieve all the annotations, as well as the frame name
      frame_name, annotations = getFrame(frame)
      strln = frame_name

      # form the string of annotations
      for annotation in annotations:
        strln += ' ' + getAnnotation(annotation)

      # write to the file
      f.write(strln + '\n')