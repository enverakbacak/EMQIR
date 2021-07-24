#coding=utf-8
import tensorflow as tf
import cv2
import glob
import numpy as np
import sys,os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import time
import glob
import os, os.path
import re
import scipy.io
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input
import tensorflow.keras.optimizers
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import pandas as pd  
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images
from tensorflow.keras.optimizers import SGD




Y = pd.read_csv(r'/home/ubuntu/Desktop/EMIR/EDMIH/Y.csv') # labels, one hot encoded
Y.shape
print(Y.shape[:])


image_size=224
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


X = np.load('preprocessed_X.npy')
X.shape
print(X.shape)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1 ,random_state=43)


batch_size = 32
epochs = 5
hash_bits = 8

visible = Input(shape = base_model.output_shape[1:])



Flatten = Flatten()(visible)
Dense_1 = Dense(1024)(Flatten)
#batchNorm = BatchNormalization()(Dense_1)
Dense_2 = Dense(hash_bits ,activation='tanh')(Dense_1)
Dense_3 = Dense(5, activation='sigmoid')(Dense_2)
model = Model(visible, Dense_3)
print(model.summary())



sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)


'''
def c_loss(noise_1, noise_2):
    def loss(y_true, y_pred):
        return (tf.keras.losses.binary_crossentropy(y_true, y_pred) + tf.keras.losses.binary_crossentropy(noise_1, noise_2) ) 
    return loss
#model.compile(loss =  c_loss(noise_1 = tf.cast(Dense_2 > 0.5 , tf.float32 ), noise_2 = Dense_2 ),  optimizer=sgd, metrics=['accuracy'],)
'''


def c_loss_1(y_true, y_pred):
    return  tf.keras.losses.binary_crossentropy(y_true, y_pred)

def c_loss_2(noise_1, noise_2):
    noise_1 = tf.cast(Dense_2 > 0.5 , tf.float32 )
    noise_2 = Dense_2 
    return  tf.keras.losses.binary_crossentropy(noise_1, noise_2)
model.compile(loss = [c_loss_1, c_loss_2],  optimizer=sgd, metrics=['accuracy'],)



history = model.fit(X_train, Y_train, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(X_valid, Y_valid) )



model_json = model.to_json()
with open("models/emqir_64_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/emqir_64_weights.h5")


params = {'legend.fontsize': 36,
          'legend.handlelength': 4,}
plt.rcParams.update(params)

matplotlib.rcParams.update({'font.size': 36})
plt.plot(history.history['accuracy'] , linewidth=5, color="green")
plt.plot(history.history['val_accuracy'], linestyle='--',  linewidth=5, color="red")
#plt.title('model accuracy' , fontsize=32)
plt.ylabel('Accuracy' , fontsize=40)
plt.xlabel('The number of epochs' , fontsize=36)
plt.legend( ['train', 'validation'], loc='best')
plt.show()
# summarize history for loss
matplotlib.rcParams.update({'font.size': 36})
plt.plot(history.history['loss'], linewidth=5, color="green")
plt.plot(history.history['val_loss'], linestyle='--', linewidth=5, color="red")
#plt.title('model loss' , fontsize=32)
plt.ylabel('Loss' , fontsize=40)
plt.xlabel('The number of epochs' , fontsize=36)
plt.legend( ['train', 'validation'], loc='best')
plt.show()


score = model.evaluate(X_valid, Y_valid, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate(X_train, Y_train)
print(model.metrics_names)
print(score)

score = model.evaluate(X_valid, Y_valid)
print(model.metrics_names)
print(score)

score = model.evaluate(X, Y)
print(model.metrics_names)
print(score)

