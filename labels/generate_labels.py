'''
This script will generate the text label files
'''

import os
import argparse

# load the utility functions
from label_utils import json2txt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, default='dataset/images_thermal_train/', help='directory of the index.json')
    parser.add_argument('--text-path', type=str, default='annotations', help='directory to save the text file')
    parser.add_argument('--name', type=str, default='', help='name of the text file')

    return parser.parse_args()


if __name__ == "__main__": 

  args = parse_args()  

  # json path
  json_path = os.path.join(args.json_path, 'index.json')

  # create text directory
  os.makedirs(args.text_path, exist_ok = True)

  json2txt(json_path, os.path.join(args.text_path, args.name + ".txt"))                  