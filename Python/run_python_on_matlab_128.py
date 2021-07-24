
# Run this on matlab!  gnome-terminal  -- python /home/ubuntu/keras/enver/ml512_finetune_loss_12/matlab_test.py
# This is only work when Matlab started from a gnome terminal in Ubuntu!

# https://www.pyimagesearch.com/2014/06/02/opencv-load-image/


import cv2
import keras
import numpy as np
from keras.applications import VGG16
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images

image_1 = plt.imread("Python/q1.jpg")
image_1 = np.array(image_1)    # converting list to array
image_1 = resize(image_1, preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
image_1 = np.array(image_1)

image_2 = plt.imread("Python/q2.jpg")
image_2 = np.array(image_2)    # converting list to array
image_2 = resize(image_2, preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
image_2 = np.array(image_2)

image_3 = plt.imread("Python/q3.jpg")
image_3 = np.array(image_3)    # converting list to array
image_3 = resize(image_3, preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
image_3 = np.array(image_3)

from keras.applications.vgg16 import preprocess_input

image_1 = preprocess_input(image_1, mode='tf')      # preprocessing the input data
image_1 = np.reshape(image_1, (1, 224, 224, 3))

image_2 = preprocess_input(image_2, mode='tf')      # preprocessing the input data
image_2 = np.reshape(image_2, (1, 224, 224, 3))

image_3 = preprocess_input(image_3, mode='tf')      # preprocessing the input data
image_3 = np.reshape(image_3, (1, 224, 224, 3))



image_size=224
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

batch_size=64
image_1 = base_model.predict(image_1, batch_size=batch_size, verbose=0, steps=None)
image_2 = base_model.predict(image_2, batch_size=batch_size, verbose=0, steps=None)
image_3 = base_model.predict(image_3, batch_size=batch_size, verbose=0, steps=None)


#coding=utf-8
from keras.models import Sequential,Input,Model,InputLayer
from keras.models import model_from_json
from keras.models import load_model
import numpy as np    # for mathematical operations

np.set_printoptions(linewidth=1024)


json_file = open('/home/ubuntu/keras/enver/dmlh2/models/dmlh2_128_model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("/home/ubuntu/keras/enver/dmlh2/models/dmlh2_128_weights.h5")
model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)

features_1 = model.predict(image_1, batch_size=64, verbose=0, steps=None)[0] #If we  do not use [0], output will be [[ ... ]]
features_2 = model.predict(image_2, batch_size=64, verbose=0, steps=None)[0] 
features_3 = model.predict(image_3, batch_size=64, verbose=0, steps=None)[0]

np.savetxt('Python/f1.txt', features_1,  fmt='%f',  delimiter=',')
np.savetxt('Python/f2.txt', features_2,  fmt='%f',  delimiter=',')
np.savetxt('Python/f3.txt', features_3,  fmt='%f',  delimiter=',')


binary_codes_1 = features_1 > 0.5
binary_codes_1 = binary_codes_1.astype(int)

binary_codes_2 = features_2 > 0.5
binary_codes_2 = binary_codes_2.astype(int)

binary_codes_3 = features_3 > 0.5
binary_codes_3 = binary_codes_3.astype(int)

np.savetxt('Python/q1.txt', binary_codes_1,  fmt='%d',  delimiter=',')
np.savetxt('Python/q2.txt', binary_codes_2,  fmt='%d',  delimiter=',')
np.savetxt('Python/q3.txt', binary_codes_3,  fmt='%d',  delimiter=',')














