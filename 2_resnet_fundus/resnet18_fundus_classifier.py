#Mount the Notebook to Drive to Access Files
from google.colab import drive
drive.mount('/content/gdrive')

#Imports libraries needed
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2 as cv2
import os as os
import h5py as h5py
import datetime
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm #gives the status of a loop (important if we have large amounts of data and need to see the progress)

#Tensorflow Imports
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator #used for data augmentation
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

def res_net_block(input_data, filters, conv_size, stride):

  if stride == 1:
      shortcut = input_data
  else:
      shortcut = Conv2D(filters, 1, strides=(stride, stride),
                        padding='same')(input_data)

  x = Conv2D(filters, conv_size, activation=None,
             padding='same', strides=(stride, stride))(input_data)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = BatchNormalization()(x)

  x = Add()([shortcut, x])
  x = Activation('relu')(x)
  return x

def resnet18():
  num_classes = 7
  filters = [64, 128, 256, 512]
  activation = 'sigmoid' if num_classes == 1 else 'softmax'
  image = Input(shape=(224,224,3), name='INPUT_LAYER')

  conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                 padding="same", activation="relu")(image)

  max_pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

  res_block1 = res_net_block(max_pool1, filters[0], 3, 1)
  res_block2 = res_net_block(res_block1, filters[0], 3, 1)

  res_block3 = res_net_block(res_block2, filters[1], 3, 2)
  res_block4 = res_net_block(res_block3, filters[1], 3, 1)

  res_block5 = res_net_block(res_block4, filters[2], 3, 2)
  res_block6 = res_net_block(res_block5, filters[2], 3, 1)

  res_block7 = res_net_block(res_block6, filters[3], 3, 2)
  res_block8 = res_net_block(res_block7, filters[3], 3, 1)

  global_average = GlobalAveragePooling2D()(res_block8)
  outputs = Dense(7, activation=activation)(global_average)

  model = Model(inputs=image, outputs=outputs)
  return model

resnet18_model = resnet18()

#compiles the resnet model
resnet18_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), 
                       metrics=['acc'])

resnet18_model.summary()

#build the callbacks

#checkpoint callback
checkpoint_path = "/content/gdrive/My Drive/Colab Notebooks/checkpoints/checkpoints_resnet18_fundus/training_batch_7_21_2020/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=20)

resnet18_model.save_weights(checkpoint_path.format(epoch=0))

#learning rate decay callback
def lrdecay(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay  

def earlystop(mode):
  if mode=='acc':
    estop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15, mode='max')
  elif mode=='loss':
    estop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')
  return estop

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

#Loads the images from image preproccessing
training_data = np.load('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/small_augmented_training_array.npy', allow_pickle=True)

training_labels = np.load('/content/gdrive/My Drive/Colab Notebooks/compressed_image_arrays/small_augmented_training_labels.npy', allow_pickle=True)

#one hot encode the labels so that their dimensions fit the output
def one_hot_encode(labels):
  x = 0 #counter
  for y in labels: #assigns a number to a label
    if y =='cataract':
      labels[x] = 0
    if y == 'glaucoma':
      labels[x] = 1
    if y == 'diabetic_retinopathy_1':
      labels[x] = 2
    if y == 'diabetic_retinopathy_2':
      labels[x] = 3
    if y == 'diabetic_retinopathy_3':
      labels[x] = 4
    if y == 'diabetic_retinopathy_4':
      labels[x] = 5
    if y == 'normal':
      labels[x] = 6
    x = x+1
  new_labels = to_categorical(labels) #one hot encodes the labels
  return new_labels

training_labels = one_hot_encode(training_labels)

#Create ImageDataGenerator for real-time augmentation
#We only augment the training data because we train with that data
train_data_gen = ImageDataGenerator(zoom_range=0.3, 
                                   width_shift_range=0.2, 
                                   height_shift_range = 0.2, 
                                   rotation_range=30)
                                   #horizontal_flip=True,
                                   #vertical_flip=True,

training_data = train_data_gen.flow(training_data, training_labels, batch_size = 32)

batch_size = 32
image_set_size = 1680
resnet_train = resnet18_model.fit_generator(training_data, 
                                              epochs=160, 
                                              steps_per_epoch= image_set_size / batch_size, 
                                              callbacks=[cp_callback, lrdecay, tensorboard_callback],)

resnet18_model.save('/content/gdrive/My Drive/Colab Notebooks/resnet18_model')
