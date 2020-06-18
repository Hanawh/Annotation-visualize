# Annotation-visualize
You can use this script to visualize the  annotations for object detection

## COCO format : json
```python
 python visualize_coco_annotation.py YOUR_DIR_TO_DATA.json YOUR_DIR_TO_DATA [random number u want to visualize]
 # eg. python visualize_coco_annotation.py /home/wh/annotations/instances_train.json /home/wh/train 10
 # It will create a directory which includes 10 images with annotations which are selected randomly
```

## VOC format : xml
```python
 python visualize_voc_annotation.py YOUR_DIR_TO_XMl YOUR_DIR_TO_DATA [random number u want to visualize]
 # eg. python visualize_coco_annotation.py /home/wh/voc2007/Annotations /home/wh/voc2007/JPEGImages train 10
 # It will create a directory which includes 10 images with annotations which are selected randomly
```
