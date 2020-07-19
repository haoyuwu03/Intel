import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

#Load in test image
path = r'C:\Users\ticto\Documents\Programming Projects\Intel\image\cat.jpg'
img = cv2.imread(path)

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

bright_img = brightness(img, 1.1, 1.5)
RGB_img = cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB) #for plt.imshow bc matplotlib default is rgb
plt.figure()
plt.subplot(221)
plt.imshow(RGB_img)

hor_flip_img = flip(img, 1)
RGB_img = cv2.cvtColor(hor_flip_img, cv2.COLOR_BGR2RGB)
plt.subplot(222)
plt.imshow(RGB_img)

vert_flip_img = flip(img, 0)
RGB_img = cv2.cvtColor(vert_flip_img, cv2.COLOR_BGR2RGB)
plt.subplot(223)
plt.imshow(RGB_img)

rotate_img = rotation(img, 100)
RGB_img = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2RGB)
plt.subplot(224)
plt.imshow(RGB_img)
plt.show()