#imports 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from tensorflow.keras.regularizers import l2

#imports from other files
from utils import read_classes
from configurations import *

#create the batch normalization class
class BatchNormalization(BatchNormalization):
    #training = false because we do not want the layer to be trainable (freezing a layer)
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

#create the convolutional layers function, there are two types of conv layers used in YOLO
def convolutional_block(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        conv = LeakyReLU(alpha=0.1)(conv)

    return conv

#create the residual block layer function (residual block like in resnet)
def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional_block(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional_block(conv       , filters_shape=(3, 3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output

#create the upsample layer function (uses feature maps from the previous layers)
def upsample_block(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

#create the darknet53 implementation function, the CNN backbone of the object detection
def darknet53_architecture(input_data):
    input_data = convolutional_block(input_data, (3, 3,  3,  32))
    input_data = convolutional_block(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)

    input_data = convolutional_block(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)

    input_data = convolutional_block(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional_block(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional_block(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

#create the YOLO function that holds the basic architecture for the object detection class
def yolov3_architecture(input_layer, NUM_CLASS):
    #after the input layer enters the Darknet-53 network, we get three branches
    route_1, route_2, conv = darknet53_architecture(input_layer)
    #see the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
    conv = convolutional_block(conv, (1, 1, 1024,  512))
    conv = convolutional_block(conv, (3, 3,  512, 1024))
    conv = convolutional_block(conv, (1, 1, 1024,  512))
    conv = convolutional_block(conv, (3, 3,  512, 1024))
    conv = convolutional_block(conv, (1, 1, 1024,  512))
    conv_lobj_branch = convolutional_block(conv, (3, 3, 512, 1024))
    
    #conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255] 
    conv_lbbox = convolutional_block(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional_block(conv, (1, 1,  512,  256))
    #upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    #upsampling process does not need to learn, thereby reducing the network parameter  
    conv = upsample_block(conv)

    conv = tf.concat([conv, route_2], axis=-1)
    conv = convolutional_block(conv, (1, 1, 768, 256))
    conv = convolutional_block(conv, (3, 3, 256, 512))
    conv = convolutional_block(conv, (1, 1, 512, 256))
    conv = convolutional_block(conv, (3, 3, 256, 512))
    conv = convolutional_block(conv, (1, 1, 512, 256))
    conv_mobj_branch = convolutional_block(conv, (3, 3, 256, 512))

    # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
    conv_mbbox = convolutional_block(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional_block(conv, (1, 1, 256, 128))
    conv = upsample_block(conv)

    conv = tf.concat([conv, route_1], axis=-1)
    conv = convolutional_block(conv, (1, 1, 384, 128))
    conv = convolutional_block(conv, (3, 3, 128, 256))
    conv = convolutional_block(conv, (1, 1, 256, 128))
    conv = convolutional_block(conv, (3, 3, 128, 256))
    conv = convolutional_block(conv, (1, 1, 256, 128))
    conv_sobj_branch = convolutional_block(conv, (3, 3, 128, 256))
    
    # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
    conv_sbbox = convolutional_block(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS +5)), activate=False, bn=False)
        
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def create_yolov3(input_size=416, channels=3, training=False, CLASSES=yolo_default_classes):
    NUM_CLASS = len(read_classes(CLASSES))
    input_layer  = Input([input_size, input_size, channels])
    conv_tensors = yolov3_architecture(input_layer, NUM_CLASS)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training: output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    YoloV3 = tf.keras.Model(input_layer, output_tensors)
    return YoloV3

#create the decode function to decode the channel information of the feature map <- idkk
def decode(conv_output, NUM_CLASS, i=0):
    # where i = 0, 1 or 2 to correspond to the three grid scales  
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position     
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
    conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
    conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box 

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size,dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * strides[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

    # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

#create the GIoU function which decides how to optimize the bounding box
def bbox_giou(boxes1, boxes2):
    ...
    # Calculate the iou value between the two bounding boxes     
    iou = inter_area / union_area 
    
    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface     
    enclose_left_up = tf.minimum (boxes1 [..., :2], boxes2 [..., :2])    
    enclose_right_down = tf.maximum (boxes1 [..., 2:], boxes2 [..., 2:])    
    enclose = tf.maximum(enclose_right_down-enclose_left_up, 0.0 ) 
    
    # Calculate the area of the smallest closed convex surface C     
    enclose_area = enclose [..., 0 ] * enclose [..., 1 ] 

    # Calculate the GIoU value according to the GioU formula     
    giou = iou- 1.0 * (enclose_area-union_area) / enclose_area 

    return giou

#create the loss calculation function
def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=yolo_default_classes):
    NUM_CLASS = len(read_classes(CLASSES))
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # Find the value of IoU with the real box The largest prediction box
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < YOLO_IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss