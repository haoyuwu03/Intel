import tensorflow as tf
import cv2
from tensorflow import keras

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Define prediction classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Function to turn image into input array
def preprocess(image_path):
    image= keras.preprocessing.image.load_img(image_path)
    input_arr = keras.preprocessing.image.img_to_array(image)
    print(input_arr[20])
    new_arr = np.resize(input_arr, (28,28)) #make same size as training images
    new_arr = np.array([new_arr])  # Convert single image to a batch.
    return new_arr

#Create input array
img = preprocess('C:/Users/ticto/Documents/Programming Projects/Intel/image/shirtwperson.jpg')


#Load model
model = tf.keras.models.load_model("clothesClassifier.model")

#Make prediction
predictions = model.predict(img)

#Round prediction to closest class and print string class name
print(class_names[np.argmax(predictions[0])])