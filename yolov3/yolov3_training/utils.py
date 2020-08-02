#get imports
import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from yolov3_training.configs import *

#clahe preproccessing function for fundus images
#credit: A. W. Setiawan, T. R. Mengko, O. S. Santoso and A. B. Suksmono, 
#"Color retinal image enhancement using CLAHE," International Conference on ICT for Smart Society, Jakarta, 
#2013, pp. 1-3. doi: 10.1109/ICTSS.2013.6588092
#implementation: Thomas Chia, Cindy Wu https://github.com/haoyuwu03/Intel/blob/master/image_preprocessing/clahe_g_channel.ipynb
def clahe(image_directory, save_directory, clipLimit = 1.0, channels = 'g'):
    #loop through each image in the directory
    for image in os.listdir(image_directory):
        #join paths
        path = os.path.join(image_directory, image)
        save_path = os.path.join(save_directory, image)
        #Step 1: Channel Splitting of the BGR Image.
        old_image = cv2.imread(path)
        old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB) #convert image to RGB
        R,G,B = cv2.split(old_image) #splits the channels
        
        #Step 2: Apply ClAHE on the "g" channel of the image
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
        
        if channels is 'g': G = clahe.apply(G)
        if channels is 'b': B = clahe.apply(B) 
        if channels is 'r': R = clahe.apply(R) 

        #Step 3: Merge Image Channels
        clahe_image = cv2.merge((B, G, R))

        #save image
        cv2.imwrite(save_path, clahe_image)

def load_yolo_weights(model, weights_file):
    tf.keras.backend.clear_session() # used to reset layer names
    # load Darknet original weights to Keras model
    range1 = 75
    range2 = [58, 66, 74]
    
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' %i
            else:
                conv_layer_name = 'conv2d'
                
            if j > 0:
                bn_layer_name = 'batch_normalization_%d' %j
            else:
                bn_layer_name = 'batch_normalization'
            
            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'

#read the class names
def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

#resizes and pads down the image so that it will pass through the network
def image_preprocess(image, target_size, gt_boxes=None):
    #the dimensions for the target size
    ih, iw    = target_size
    #actual scale of the image, includes pixel channels
    h,  w, _  = image.shape
    #scales down the image
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    #resizes the image based on the scaling 
    image_resized = cv2.resize(image, (nw, nh))

    #pads the image to prevent distortion
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    #scales down the pixel values
    image_paded = image_paded / 255.

    #used if there are no bounding boxes, just returns image
    if gt_boxes is None:
        return image_paded

    #if there are bounding boxes, then will rescales bounding boxes accordingly
    else:
        #formula: bounding boxes = (bounding box * the scale) + the changed dimensions
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


#define bbox display function
def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):   
    #find the number of classes
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    #determine image size: h,w,channels
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    #determine colors to use in the display when drawing the bounding boxes
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    #shuffle the colors to be used
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    #loop through each bounding box
    for i, bbox in enumerate(bboxes):
        #determine bounding box coordinates, which are the first four values in bbox
        coor = np.array(bbox[:4], dtype=np.int32)
        #the score aka the prediction is the fifth value
        score = bbox[4]
        class_ind = int(bbox[5])
        #determine the bounding box color, colors are different per class
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        #determine the thickness of the bounding box
        if bbox_thick < 1: bbox_thick = 1
        #define the size of the font as a scale factor of the bounding box size
        fontScale = 0.5 * bbox_thick
        #determine the coordinates
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        #print out the object rectangle
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


#determine the intersection over union of the bounding boxes
def bboxes_iou(boxes1, boxes2):
    #define the boxes
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    #determine the area of the boxes
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    #determine the difference of the area
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    #calculate the iou of the bounding boxes
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

#define the non-maximum supression function to only limit the bounding boxes that we want
def nms(bboxes, iou_threshold, sigma=0.3):
    #find the classes in the bounding box array
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    #loop through each class in the classes
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
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

#define the post processing function
def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    #1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    #change the format to coordinates instead of high, width and original pixel values
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

#detects objects based on images
def detect_image(model, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', colab = True):
    #reads original image
    original_image      = cv2.imread(image_path)
    #converts image to rgb format
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    #preprocess the images so they are in the correct format
    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    #use clahe on g channel
    #image_data = clahe(image_data, './dataset/')
    #expand the dimensions
    image_data = tf.expand_dims(image_data, 0)

    #predict the bounding box using the model
    pred_bbox = model.predict(image_data)
    #reshape each prediction per bounding box
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    #post process the boxes into a usable format
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    #apply non maximum supression to the boxes
    bboxes = nms(bboxes, iou_threshold)

    #draw the boxes and labels on the image
    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

    #save the image
    if output_path != '': cv2.imwrite(output_path, image)

    #display the image depending on requirements
    if (show == True) and (colab == True):
        #cv2.imshow does not work in google colab
        plt.imshow(image)
        
    elif (show == True) and (colab == False):  
        #show the image
        cv2.imshow("predicted image", image)
        #load and hold the image
        cv2.waitKey(0)
        #to close the window after the required kill value was provided
        cv2.destroyAllWindows()
        
    return image

#detects objects based on videos
def detect_video(model, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    times = []
    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    while True:
        _, img = vid.read()

        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = tf.expand_dims(image_data, 0)
        
        t1 = time.time()
        pred_bbox = model.predict(image_data)
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')
        
        times.append(t2-t1)
        times = times[-20:]
        
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        
        print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

        image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()

#detect from webcam
def detect_realtime(YoloV3, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    times = []
    vid = cv2.VideoCapture(0)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        image_frame = image_preprocess(np.copy(original_frame), [input_size, input_size])
        image_frame = tf.expand_dims(image_frame, 0)
        
        t1 = time.time()
        pred_bbox = YoloV3.predict(image_frame)
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')
        
        times.append(t2-t1)
        times = times[-20:]
        
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        
        print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

        frame = draw_bbox(original_frame, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        image = cv2.putText(frame, "Time: {:.1f}FPS".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if output_path != '': out.write(frame)
        if show:
            cv2.imshow('output', frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
