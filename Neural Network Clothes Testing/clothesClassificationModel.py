import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#Load data from keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Label names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Scale pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

#Build Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #turns 2d array to 1d
    keras.layers.Dense(128, activation='relu'), #every node connected to next
    keras.layers.Dense(10), #Each output node is score of input belonging to one of the 10 classes
    keras.layers.Softmax()
])

#Compile Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train Model
model.fit(train_images, train_labels, epochs=10)

#Evaluate model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#Make probability prediction model
#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

#Make predictions
#predictions = probability_model.predict(test_images)

#Find largest probability
#print(np.argmax(predictions[0]))

#Save probability_model
model.save('clothesClassifier.model')