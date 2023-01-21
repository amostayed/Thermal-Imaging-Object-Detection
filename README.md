# Introduction
<p align="center">
  <img src="/images/gif/demo.gif" alt="animated" width="512" height="384"/>
</p>

A PyTorch implementation of the [YOLOv5](https://github.com/ultralytics/yolov5) model for thermal object detection. This repository builds a medium YOLOv5 model on the [FLIR ADAS v2.0](https://www.flir.com/oem/adas/adas-dataset-form/) dataset. This implementation uses a single detection head across scales unlike YOLO v3-v5.   

## Data Preparation
- [Download](https://www.flir.com/oem/adas/adas-dataset-form/) the FLIR thermal dataset.
- Unzip the contents of the zip file and put them in the `dataset` folder.
- The training and validation images for thermal modality should reside in `images_thermal_train` and `images_thermal_val` respectively.
- Prepare the label files:
  - Convert the index.json files to text annotation files that will be used by the PyTorch Dataset class
  - Move to `labels` directory and run ``` python generate_labels.py --json-path "../dataset/images_thermal_train" --text-path "annotations" --name "annotations_thermal_train" ``` to create the text file for training images inside the `annotations` sub-folder.\
  - Do the same for the validation & test sets.
  - For custom file names remember to change the configuration yaml file (see Training section) accordingly.  

- Anchor box clustering:
  - Move to `preprocessing` and run the notebook.
  - The generated anchor boxes should be entered in the yaml file for training later.

## Training
- Modify the `config.yml` file inside `configurations`.
- Remember to change the configurations accordingly for custom file names and anchor box clustering.
- Note: anchor boxes are entered largest scale to smallest in the yaml file
- Two pre-trained models are provided in the `pretrained models` folder:
  - `coco.pth` : YOLOv5m pretrained on COCO (from [YOLOv5 source](https://github.com/ultralytics/yolov5))
  - `cityscape.pth` : which is a model trained by me on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. By default, this model will be loaded for FLIR training.
  - Set `load from` key to `null` in the configuration file for training from scratch
  - run `python training.py --config-path ./configuration/config.yml` from the root directory (enter your custom config file)
  - The trained model will be saved in `saved models`

## Inference
- Two notebooks inside the `inference` folder to generate AP and detection results for the video test set.
- They are self-explanatory.
- A trained model is provide in the `saved models` folder.
- AP for the trained model:

  | Model  | Person  | Car |
  | :------------ |:---------------:| -----:|
  | YOLOv5      | **81.26** | **79.28** |
  | FLIR Baseline (YOLOX)      | 75.33        |   75.23 |
