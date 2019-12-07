#!/usr/bin/env python3
"""A YOLO objection detection model class"""


import tensorflow.keras as K
import tensorflow as tf


class Yolo:
    """A YOLO object detection class
    model: the YOLO model
    classes: the class labels for categorization
    class_t: threshold for object detection in boxes for initial filter step
    nms_t: the IoU threshold for non-max suppression
    anchors: anchor boxes"""
    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path) as class_file:
            self.class_names = [line.rstrip('\n') for line in class_file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
