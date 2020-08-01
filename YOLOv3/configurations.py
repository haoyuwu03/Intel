#imports
import numpy as np

#the yolo default configuration options
yolo_darknet_weights = "D:\YOLOv3\yolov3_configs\yolov3.weights"
yolo_custom_weights = False
yolo_default_classes = "D:\YOLOv3\yolov3_configs\coco.names"
yolo_strides = [8, 16, 32]
yolo_IOU = 0.5 #https://pjreddie.com/media/files/papers/YOLOv3.pdf Section 5
yolo_anchor_perscale = 3
yolo_boundbox_max_perscale = 100
yolo_input_size = 416 #images are sizes of 416x416
yolo_anchors = [[[10,  13], [16,   30], [33,   23]], #anchors allow sections to have more than one bounding boxes
                [[30,  61], [62,   45], [59,  119]], #https://towardsdatascience.com/you-only-look-once-yolo-implementing-yolo-in-less-than-30-lines-of-python-code-97fb9835bfd2#:~:text=What%20are%20anchor%20boxes%3F,cell%20to%20detect%20multiple%20objects.
               [[116, 90], [156, 198], [373, 326]]]

strides = np.array(yolo_strides)
anchors = (np.array(yolo_anchors).T/strides).T

#training options
train_classes = "path_to_custom_classes"
train_annots = "path_to_annotations"
train_logs = "path_to_logs"
train_checkpoints = "path_to_checkpoints_folder"
train_bsize = 8 #batch size
train_input_size = 416 #image input size
train_start_lr = 1e-4 #starting learning rate
train_end_lr = 1e-6 #ending learning rate
train_epochs = 100 #number of epochs
train_augmentation = True #use data augmentation
train_transfer = True #use transfer learning
save_checkpoint = True #save the training checkpoints
use_checkpoints = False #continue training off of prev. checkpoint

#testing options
test_annotations = "path_to_test_annotations"
test_detected_image = "path_to_directory"
test_bsize = 4 #batchsize for the testing
test_input_size = 416 #image input size
test_threshold = 0.3
test_iou = 0.45
test_augmentation = False #using data augmentation


