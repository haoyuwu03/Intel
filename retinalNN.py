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

IMG_SIZE = 150
#path to the training file
data_directory = "C:/Users/ticto/.kaggle/train1"
test_directory = "C:/Users/ticto/.kaggle/test1/test" #unused during model creation
data_labels_df = pd.read_csv("C:/Users/ticto/.kaggle/new_train_labels.csv")

#List of image filenames
files = os.listdir(data_directory)

#Ordered list of image filenames
data_files = []

#Order list of image filenames by training label dataframe
for filename in data_labels_df["image"]:
    name = filename + ".jpeg"
    #try:
    file_index = files.index(name)
    data_files.append(files[file_index])
    #except:
        #train_labels_df = train_labels_df[train_labels_df.image != filename] #getting rid of training label dataframe rows of non-existant images

#train_labels_df.to_csv('C:/Users/ticto/.kaggle/new_train_labels.csv') #saving updated training label dataframe

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
def color_crop_enhance(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX),-4,128)
    return image

#Display 10 images
def plotImages(images_arr):
    fig, axes = plt.subplots(2, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#Create image array
def create_input_data(folder_path, filename_list):
    data = []
    for img in filename_list:
        image = color_crop_enhance(os.path.join(folder_path,img))
        data.append(image)
    return np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

#Create array of train images
train_images = create_input_data(data_directory, data_files[0:40]) #takes too long to preprocess all the images

#Create array of test images
test_images = create_input_data(data_directory, data_files[40:50]) #takes too long to preprocess all the images

#Scale pixel values
train_images = train_images / 255.0

#Create array of training and testing labels
data_labels = np.array(data_labels_df['level'])

train_labels = data_labels[0:40]
test_labels = data_labels[40:50]

#Build Model
model = keras.models.Sequential([
    keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=train_images.shape[1:]),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(3), #Each output node is a score of the three classes
    keras.layers.Softmax()
])

#Compile Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train Model
model.fit(train_images, train_labels, batch_size=2, epochs=3)

#Evaluate Model on Training set
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#Save Model
model.save('firstRetinalNN.model')

