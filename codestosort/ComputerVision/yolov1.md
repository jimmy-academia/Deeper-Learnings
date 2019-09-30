# Yolo v1

> Building Yolo v1 with VGG and ResNet backbone for Aerial Images. Came across this problem in HW2 of Deep Learning Computer Vision course 

## Settings
* The Dataset: [DOTA: A Large-scale Dataset for Object DeTection in Aerial Images](http://captain.whu.edu.cn/DOTAweb/index.html)
* The Model: VGG/ResNet backbone for yolo v1
* The goal: MAP score > 8.6%

## Introductions

#### YOLO v1
Yolo (you only look once) is a Deep Learning Design for object detection. A normal intuition for object detection would be to recurrently feed the image through a deep model system to narrow down the region of interest (as is the case for [RCNN](https://paperswithcode.com/paper/fast-r-cnn)). Yolo on the other hand is designed to output the correct bounding boxes for every object appearing in an image by passing the image throught the system once. This is done by outputing a huge output vector of size (Batch Size, S, S, (B\*5+C)) representing the multiple guesses for predicting a bounding box. 
Detail of the output vector is:
```
    in S x S equally divided cells:
        predict B boxes, each having a set of (center x, center y, width, height, confidence)
        along with an one hot vector of length C (total number of classes)
```

#### MAP score
For any prediction, a table of four componets can be drawn:
| . \      | predict True   | predict False |
| ------------------------------------------|
| is True  | True positive  | False negative|
| ---------| ---------------| --------------|
| is False | False positive | True negative |

And from these 4 values we can also calculate 

The prediction of yolo comes with a confidence level (0-1). If we filter the prediction outcomes according to confidence level, we would get all (with threshold=0) or none (with threshold=1) and 