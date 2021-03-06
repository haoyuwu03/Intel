import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

#Mount the Notebook to Drive to Access Files
from google.colab import drive
drive.mount('/content/gdrive')

#Loads the images from image preproccessing
training_data = np.load('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/small_training_array.npy', allow_pickle=True)
#validation_data = np.load('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/augmented_validation_array.npy', allow_pickle=True)

training_labels = np.load('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/small_training_labels.npy', allow_pickle=True)
#validation_labels = np.load('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/augmented_validation_labels.npy', allow_pickle=True)

def brightness(img, low, high): #pass in range where random value is chosen
    value = random.uniform(low, high) #choose random value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert bgr to hsv
    hsv = np.array(hsv, dtype = np.float64) #turn into float array
    hsv[:,:,1] = hsv[:,:,1]*value #brighter if value is > 1
    hsv[:,:,1][hsv[:,:,1]>255]  = 255 #set cap of 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8) #back to int
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) #back to BGR
    return img

def flip(img, flip_direction):
    return cv2.flip(img, flip_direction)

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

#function for augmenting images, 4x more images
def augment_images(image_array, image_labels):
  unaugmented_images = image_array
  old_labels = image_labels
  augmented_images = []
  augmented_labels = []

  counter = 0 #counter used so that each of the images will have appropriate labels
  for old_image in unaugmented_images:
    horizonatal_im = flip(old_image, 1) #flips horizonatally
    vertical_im = flip(old_image, 0) #flips vertically
    horizontal_vertical_im = flip(horizonatal_im, 0) #flips horizonatally then vertically
    normal_im = old_image #keep original image

    for label in range(4):
      add_label = old_labels[counter] #looks for the correct label for the image
      augmented_labels.append(add_label) #adds label to new updated label list

    augmented_images.extend((horizonatal_im, vertical_im, horizontal_vertical_im, normal_im)) #adds the augmented images to the new array

    counter = counter + 1 #updates counter

  return np.asarray(augmented_images), np.asarray(augmented_labels)

augmented_images, labels = augment_images(training_data, training_labels)

print(labels.shape)

print(augmented_images.shape)

np.save('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/small_augmented_training_array.npy', augmented_images) #saves the numpy image arrays

np.save('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/small_augmented_training_labels.npy', labels) #saves the numpy label arrays
