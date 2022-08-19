#import libraries
import tensorflow.keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import keras
from keras import models
from keras import layers
from keras.layers.core import Permute
import tensorflow as tf
import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
tf.enable_eager_execution() 

from PIL import Image
import glob
import cv2
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import os
import matplotlib.pyplot as plt
%matplotlib inline

direct = r'C:/../Dataset/' #identify directory of image folder path
classes = [clas.name for clas in os.scandir(direct) if clas.is_dir()] #assign labels to the images based on its folder name

X, y = [], []
for clas in classes:
    path = os.path.join(direct, clas)
    for img in os.listdir(path):
        X.append(cv2.imread(os.path.join(path, img)))
        y.append(int(clas))
        
X = X.reshape(4339,3,224,224).transpose(0,2,3,1).astype("uint8") #change 4339 to the number of images
y = np.array(y)

def one_hot_encode(vec, vals = 2): #define based on class number
    #to one-hot encode the 4- possible labesl
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out
  
class CifarHelper():
    def __init__(self):
        self.i = 0
        self.images = None
        self.labels = None
    def set_up_images(self):
        print("Setting up images and labels")
        self.images = np.vstack([X])
        all_len = len(self.images)
        self.images = self.images.reshape(4339, 3, 224, 224).transpose(0,2,3,1)/255 #change number accordingly
        self.labels = one_hot_encode(np.hstack([y]), 2)

#before tensorflow run:
#set up images with labels
ch = CifarHelper()
ch.set_up_images()

#encode labels
def to_one_hot(y, dimension=2):
    results = np.zeros((len(y), dimension))
    for i, label in enumerate(y):
        results[i, label] = 1.
    return results

one_hot_labels = to_one_hot(y)

def load_and_preprocess_from_path_label(X, y):
    X = 2*tf.cast(X, dtype=tf.float32) / 255.-1
    y = tf.cast(y, dtype=tf.int32)

    return X, y
  
#import libraries for training, testing split  
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

#training, validation, testing split with 8:1:1 rule
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2, random_state=42, shuffle = True)
X_validation, X_test, y_validation, y_test = train_test_split(X_validation, y_validation, test_size = 0.5, random_state=42, shuffle = True)

from tensorflow.keras import layers, Model, Sequential, regularizers

#construct individual learner - Xception
def entry_flow(inputs) :

    x = Conv2D(32, 3, strides = 2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64,3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation = x

    for size in [128, 256, 728] :

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding='same')(x)

        residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)

        x = tensorflow.keras.layers.Add()([x, residual])
        previous_block_activation = x

    return x
  
def middle_flow(x, num_blocks=8) :

    previous_block_activation = x

    for _ in range(num_blocks) :

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = tensorflow.keras.layers.Add()([x, previous_block_activation])
        previous_block_activation = x

    return x
  
def exit_flow(x) :

    previous_block_activation = x

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x) 
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=2, padding='same')(x)

    residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = tensorflow.keras.layers.Add()([x, residual])

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='linear')(x)

    return x  

inputs = Input(shape=(224,224,3))
outputs = exit_flow(middle_flow(entry_flow(inputs)))
xception = Model(inputs, outputs)

import time
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score 
from keras.callbacks import ModelCheckpoint, EarlyStopping

#define check_point for model training
checkpoint_filepath = 'weights.{epoch:02d}-{val_loss:.2f}.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                            monitor = 'val_accuracy',
                                                            mode = 'max',
                                                            save_best_only=True)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics = ['accuracy'])

#model fitting, this will select the best-performing model
history = model.fit(X_train, y_train, epochs = 30,
                   validation_data=(X_validation, y_validation), batch_size = 2, callbacks=[model_checkpoint_callback])

model.load_weights('weights.11...h5') #select the best-performing model in the folder with the highest validation accuracy
model.save('saved_model/Model A') #save individual learner model A

from sklearn.metrics import confusion_matrix

#make predictions
test = model.predict(X_validation, batch_size = 2)
pred = np.argmax(test, axis=1)

#evaluate the model and store its performance
confusion_matrix(y_validation, pred)
