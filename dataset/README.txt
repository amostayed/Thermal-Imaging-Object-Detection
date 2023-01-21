1. Download the FLIR v2.0 Dataset (zip file) from here: https://www.flir.com/oem/adas/adas-dataset-form/
2. Unzip and put the following folders here (rgb images are optional):
   images_rgb_train 
   images_rgb_val 
   images_thermal_train 
   images_thermal_train 
   video_rgb_test 
   video_rgb_test 
3. (optinal if you are doing fusion experiments) Also download the rgb-thermal frame mapping (images_rgb_train.json)and put here
4.  A note on the annotations: index.json in each folder is a format produced by FLIR/Teledyne data curation software Conservator. A corresponding clean
    MS-COCO version is also availabel (coco.json)
5. Also read readme.txt from the FLIR download
