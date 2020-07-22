#Imports libraries needed
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2 as cv2
import os as os
import h5py as h5py
from tqdm import tqdm #gives the status of a loop (important if we have large amounts of data and need to see the progress)

#Mount the Notebook to Drive to Access Files
from google.colab import drive
drive.mount('/content/gdrive')

#Link all directories
image_directory = "/content/gdrive/My Drive/image_directory" #path to main directory


#Paths to directories
files = os.listdir(image_directory)
training_images = os.path.join(image_directory, "training_images/") #path to training images
validation_images = os.path.join(image_directory, "validation_images/") #path to validation images
testing_images = os.path.join(image_directory, "testing_images/") #path to testing_images


#File names
image_labels = ['cataract', 'hypertensive_retinopathy', 'glaucoma', 'diabetic_retinopathy_1', 
                'diabetic_retinopathy_2', 'diabetic_retinopathy_3', 'diabetic_retinopathy_4','normal']


#Empty lists that will contain the images and labels per category
training_data = []
training_labels = []
validation_data= []
validation_labels = []
testing_data = []
testing_labels = []

training_data_size = []
validation_data_size = []
testing_data_size = []

#Code to split the images and their respective labels
def image_and_labels(directory_path, type):
  for label in image_labels:  #for each of the labels read the folder 
    path = os.path.join(directory_path, label)  #create a path to each of the folders
    #class_num = image_labels.index(label) #assigns a class number to each of the labels
    for img in tqdm(os.listdir(path)):  #iterates over each image 
      img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_ANYCOLOR)  #converts to an array
      img_array = create_input_data(img_array)

      #if else statements to add the data into the associated arrays
      if type == 'train':
        training_data.append(img_array) #adds the image array and the label (label)
        training_labels.append(label)
        #training_data_size.append('1') # only do this onetime to find out the size of the training
      elif type == 'test':
        testing_data.append(img_array) #adds the image array and the label (label)
        #testing_labels.append(label)
      elif type == 'vali':
        validation_data.append(img_array) #adds the image array and the label (label)
        validation_labels.append(label)
        #validation_data_size.append('1') # only do this one time to find the size of the validation

#Image Pre-Processing (CLAHE on "G" Channel)
#Image Pre-Processing Code is designed from "Color Retinal Image Enhancement using CLAHE" research paper.

def clahe_process(image_matrix, BGR = True): #filepath is the path to the file, BGR is if the image is BGR format
  old_image = image_matrix

  #Step 1: Splitting the R, G and B channels (after converting to RGB format if needed)
  if BGR == True: #All images that i have checked for cataracts, glaucoma, normal, and diabetic retinopathy are BGR
    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)
  else:
    old_image = image
  R, G, B = cv2.split(old_image)

  #Step 2: Apply CLAHE on the "G" Channel of the image
  clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize = (8,8)) #Creates a clahe function with clip limit of 1.0 and a comparison range of 8 x 8
  G = clahe.apply(G)

  #Step 3: Merge image channels
  new_image = cv2.merge((R, G, B))
  return new_image

#Changes the image data into CLAHE filtered data
def create_input_data(matrix_image):
    #data = []
    #for matrix_image in image_matrix_list:
        matrix_image = clahe_process(matrix_image)
        #data.append(image)
        resized_image = cv2.resize(matrix_image, (224, 224))
        return resized_image/255.0 #to rescale th image

#saving huge numpy arrays: http://chrisschell.de/2018/02/01/how-to-efficiently-deal-with-huge-Numpy-arrays.html
#DO NOT RUN# DO NOT RUN#
image_and_labels(testing_images, 'test') #changes all testing data into arrays
image_and_labels(training_images, 'train') #changes all training data into arrays
image_and_labels(validation_images, 'vali') #changes all validation data into arrays


testing_data = np.asarray(testing_data)
training_data = np.asarray(training_data)
validation_data = np.asarray(validation_data)

np.save('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/testing_array.npy', testing_data) #saves the numpy image arrays
np.save('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/small_training_array.npy', training_data) #saves the numpy image arrays
np.save('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/validation_array.npy', validation_data) #saves the numpy image arrays

testing_labels = np.asarray(testing_labels)
traning_labels = np.asarray(training_labels)
validation_labels = np.asarray(validation_labels)

np.save('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/testing_labels.npy', testing_labels) #saves the numpy label arrays
np.save('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/small_training_labels.npy', training_labels) #saves the numpy label arrays
np.save('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/validation_labels.npy', validation_labels) #saves the numpy label arrays
