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
import time

IMG_SIZE = 500
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

def clahe_image(path, clip_limit=2, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #splits image into its rgb channels
    R,G,B = cv2.split(image)

    #defines the clahe cv2 function
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))

    R = clahe.apply(R)
    G = clahe.apply(G)
    B = clahe.apply(B)

    image = cv2.merge((R, G, B))
    return(image)

def create_input_data(folder_path, filename_list):
    data = []
    for img in filename_list:
        image = clahe_image(os.path.join(folder_path,img))
        data.append(image)
    return np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

#Values
train_size = 500
val_size = 100
batch_size = 10
eopchs = 10

#Create array of train images
train_images = create_input_data(data_directory, data_files[0:train_size]) #takes too long to preprocess all the images

#Create array of validation images
val_images = create_input_data(data_directory, data_files[train_size:train_size+val_size]) #takes too long to preprocess all the images

#Scale pixel values
train_images = train_images / 255.0
val_images = val_images / 255.0

#Create array of training and testing labels
data_labels = np.array(data_labels_df['level'])

train_labels = data_labels[0:train_size]
val_labels = data_labels[train_size:train_size+val_size]

#Create ImageDataGenerator for real-time augmentation
train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, 
                                                                width_shift_range=0.1, 
                                                                height_shift_range = 0.1, 
                                                                horizontal_flip=True,
                                                                vertical_flip=True,
                                                                rotation_range=100)
val_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, 
                                                                width_shift_range=0.1, 
                                                                height_shift_range = 0.1, 
                                                                horizontal_flip=True,
                                                                vertical_flip=True,
                                                                rotation_range=100)

#Pass images and labels into ImageDataGenerator
train_data = train_DataGen.flow(train_images, train_labels, batch_size=batch_size) 
val_data = val_DataGen.flow(val_images, val_labels, batch_size=batch_size) 

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
#model.fit(train_images, train_labels, batch_size=2, epochs=3)

#Train Model
history = model.fit_generator(
    train_data,
    steps_per_epoch=train_size // batch_size,
    epochs=epochs,
    validation_data=val_data,
    validation_steps=val_size // batch_size
)

#View Results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Save Model
model.save('augmentedRetinalImagesNN.model')

os._exit(0)

#UNUSED OLD CODE
#adds file name to category
"""for filename in files_training:
    if filename[0:2] == "NL":
        labels.append(0)
        file_names.append(filename)
    if filename[22:23] == "0":
        labels.append(0)
        file_names.append(filename)
    if filename[22:23] == "1":
        labels.append(1)
        file_names.append(filename)
    if filename[22:23] == "2":
        labels.append(2)
        file_names.append(filename)
    if filename[22:23] == "3":
        labels.append(3)
        file_names.append(filename)
    if filename[22:23] == "4":
        labels.append(4)
        file_names.append(filename)
    if filename[0:4] == "Glau":
        labels.append(5)
        file_names.append(filename)
    if filename[0:3] == "cat":
        labels.append(6)
        file_names.append(filename)"""



#combines matrix in a dataframe
"""data = pd.DataFrame({
    'filename' : file_names,
    'label' : labels,
})"""

#plots the data being used
#data['label'].value_counts().plot.bar()
#print(data['label'].value_counts())
