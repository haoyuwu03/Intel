#NOTE: the process of creating training data needs to be fixed- it created a (150, 3) shaped array instead of a (150, 150) one. Look into feeding preprocessed images of type PIL into a Keras ImageDataGenerator

#imports
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import io
import cv2
import csv
import skimage.io

#path to the training file
data_directory = "C:/Users/ticto/Documents/Programming Projects/Intel/RetinopathyImages/labeled_train"
files_training = os.listdir(data_directory)

# %%
#file name and category matrix
categories = []
file_names = []

#adds file name to category
for filename in files_training:
  if filename[22:23] == "0":
    categories.append("healthy eye")
    file_names.append(filename)
  if filename[22:23] == "1":
    categories.append("diabetic rectinopathy stage 1")
    file_names.append(filename)
  if filename[22:23] == "2":
    categories.append("diabetic rectinopathy stage 2")
    file_names.append(filename)
  if filename[22:23] == "3":
    categories.append("diabetic rectinopathy stage 3")
    file_names.append(filename)
  if filename[22:23] == "4":
    categories.append("diabetic rectinopathy stage 4")
    file_names.append(filename)

#combines matrix in a dataframe
data = pd.DataFrame({
    'filename' : file_names,
    'label' : categories,
})


# %%
#plots the data being used
data['label'].value_counts().plot.bar()
print(data['label'].value_counts())

#Crop image
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark to crop
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    		#print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    		#print(img.shape)
        return img

#Enhance colors and add halo around eye
def color_crop_enhance(path, img_size, sigmaX=10):

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX),-4,128)
    return image

# %%
#visualize images, our images are BGR images (the diabetic rectinopathy ones, so colors will be flipped when being read using cv2.imread, idk if this will have a difference)
IMG_SIZE = 150

#Train images array
train_images = []
for img in os.listdir(data_directory):
    image = color_crop_enhance(os.path.join(data_directory,img), IMG_SIZE)
    train_images.append(image)
    #img_array = cv2.imread(os.path.join(data_directory,img),cv2.IMREAD_GRAYSCALE)
    
    #plt.show()
    #print(img)

#Display 10 images
def plotImages(images_arr):
    fig, axes = plt.subplots(2, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(train_images)

#Turn images into array
train_data = []
for image in train_images:
    train_data.append(tf.keras.preprocessing.image.img_to_array(image))
#print(train_data)

#Scale pixel values
for image in train_data:
    for column in image:
        for row in column:
            for pixel in row:
                pixel = pixel / 255.0

#Create array of training labels
train_labels = data['label']

#Build Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(150, 3)), #turns 2d array to 1d
    keras.layers.Dense(128, activation='relu'), #every node connected to next
    keras.layers.Dense(3), #Each output node is score of input belonging to one of the 3 classes
    keras.layers.Softmax()
])

#Compile Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train Model
model.fit(train_data, train_labels, epochs=10)

#ADD TEST DATA TO EVALUATE MODEL ON!!!

#Save Model
model.save('firstRetinalNN.model')

