#imports 
import numpy as np
import tensorflow as tf
import cv2
import time
import random
import colorsys
import matplotlib.pyplot as plt
import os
import shutil
import json
#imports from other files
from configurations import yolo_default_classes

#start of the functions list
#functions that are created here: 
#load_yolo_weights(), read_classes(), image_preprocessing(), draw_box(), box_iou(), non_max_supression(), post_process(), detect_image()

#loads the yolo_weights and 
def load_yolo_weights(model, weights_file):
    #load previous darknet weights 
    layer_range_1 = 75
    layer_range_2 = [58, 66, 74]

    #opens the weight file with temp variable
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count = 5) #skips the first 5 header values

        batch_normalization_counter = 0
        #loops through the file so that the layers and associated weights will be loaded in 
        for weight in range(layer_range_1):
            if weight > 0:
                convolutional_layer_name = "conv2d_%d" % weight
            else:
                convolutional_layer_name = "conv2d"

            if batch_normalization_counter > 0:
                batch_normalization_layer_name = "batch_normalization_%d" % batch_normalization_counter
            else: 
                batch_normalization_layer_name = "batch_normalization"

            #gets the convolutional layer so we can apply filters, kernal size and dimensions
            convolutional_layer = model.get_layer(convolutional_layer_name) #gets the layer from the model to load the weights
            filters = convolutional_layer_name.filters
            kernal_size = convolutional_layer_name.kernal_size[0]
            input_dimensions = convolutional_layer_name.input_size[-1]

            if weight not in range(layer_range_2):
                #sets the batch normalization weights
                batch_normalization_weights = np.fromfile(wf, dtype = np.float32, count = 4 * filters) #gets the batch normalization weights from the file
                batch_normalization_weights = batch_normalization_weights.reshape((4, filters))[[1, 0, 2, 3]]
                batch_normalization_layer = model.get_layer(batch_normalization_layer_name)
                batch_normalization_counter += 1
            else:
                #sets the convolutional bias
                convolutional_bias = np.fromfile(wf, dtype = np.float32, count = filters)

            #sets the convolutional layer shape
            convolutional_shape = (filters, input_dimensions, kernal_size, kernal_size)
            convolutional_weights = np.fromfile(wf, dtype = np.float32, count = np.product(convolutional_shape))

            if weight not in layer_range_2:
                convolutional_layer.set_weights([convolutional_weights])
                batch_normalization_layer.set_weights(batch_normalization_weights)
            else:
                convolutional_layer.set_weights([convolutional_weights, convolutional_bias])
        
        #alert for data not read properly
        assert len(wf.read()) == 0, 'failed to read all data'

#read the class names from the coco class name file
def read_classes(class_file_name):
    #loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    #retuns a file with all of the names 
    return names

#preprocesss images so the fit into the model
def image_preprocessing(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def draw_box(image, bboxes, classes=yolo_default_classes, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):   
    NUM_CLASS = read_classes(classes)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        #put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            #get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            label = "{}".format(NUM_CLASS[class_ind]) + score_str

            #get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            #put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            #put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image

#create the intersection over union function for the bounding boxes
def box_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

#create the non-maximum suppression function for the bounding box selection
def non_max_supression(bboxes, iou_threshold, sigma=0.3, method='nms'):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        #Process 1: Determine whether the number of bounding boxes is greater than 0 
        while len(cls_bboxes) > 0:
            #Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            #Process 3: Calculate this bounding box A and
            #Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
            iou = box_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

#bounding box post processing
def post_process(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    #1. (x, y, w, h) -> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    #2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    #3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    #4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    #5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

#create the image detection function
def detect_image(YoloV3, image_path, output_path, input_size=416, show=False, CLASSES=yolo_default_classes, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', colab = True):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocessing(np.copy(original_image), [input_size, input_size])
    image_data = tf.expand_dims(image_data, 0)

    pred_bbox = YoloV3.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = post_process(pred_bbox, original_image, input_size, score_threshold)
    bboxes = non_max_supression(bboxes, iou_threshold, method='nms')

    image = draw_box(original_image, bboxes, classes=CLASSES, rectangle_colors=rectangle_colors)
    
    #if output_path != '': 
        #cv2.imwrite(output_path, image)
        
    if show & (colab == True):
        #show the image
        plt.imshow(image) #changed from original code because cv2 does not work
    elif (show == True) & (colab == False):
        cv2.imshow('output', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image
    