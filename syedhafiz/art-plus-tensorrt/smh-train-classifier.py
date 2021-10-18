from art.attacks.evasion.carlini import CarliniL0Method, CarliniLInfMethod
import tensorflow as tf
from tensorflow.python.keras.activations import get

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization,Activation,Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import ResNet50, VGG19
import numpy as np

from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, to_categorical
from art.utils import load_cifar10

###############
import matplotlib as mpl
import os
from os import listdir
import time
from matplotlib import pyplot as plt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import convert_to_constants
import sys

import argparse


# original_stdout = sys.stdout 
# f = open("output-classifier-for-cifar10-tf.txt", "a")
# sys.stdout = f
print('# '*50+str(time.ctime()))
tmpdir = os.getcwd()
time_weight=1000
MODEL_NAME={"VGG":"VGG19", "RESNET":"ResNet50"}
# Step 1: Load the dataset
(x_train, y_train), _, min_pixel_value, max_pixel_value = load_cifar10()

n_train_samples = int(sys.argv[1])
batch_size = int(sys.argv[2])
nb_epochs= int(sys.argv[3])
keras_file_name=str(sys.argv[4])
model_name=str(sys.argv[5])

x_train = x_train[:n_train_samples]
y_train = y_train[:n_train_samples]
print((x_train.shape,y_train.shape))

model= Sequential()
# Step 2: Load Model and do transfer learning 
if model_name==MODEL_NAME.get("VGG"):
    base_model = VGG19(include_top=False,weights='imagenet',input_shape=(32,32,3),classes=y_train.shape[1])
    model.add(base_model)
    model.add(Flatten()) 
    model.add(Dense(1024,activation=('relu'),input_dim=512))
    model.add(Dense(512,activation=('relu'))) 
    model.add(Dense(256,activation=('relu'))) 
    model.add(Dense(128,activation=('relu')))
    model.add(Dense(10,activation=('softmax')))
    # print(model.summary())
    sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
    model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=["accuracy"])
elif model_name==MODEL_NAME.get("RESNET"):
    learning_rate=0.001
    base_model = ResNet50(include_top=False,weights='imagenet',input_shape=(32,32,3),classes=y_train.shape[1])
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(4000,activation=('relu'),input_dim=512))
    model.add(Dense(2000,activation=('relu'))) 
    model.add(Dropout(.4))
    model.add(Dense(1000,activation=('relu'))) 
    model.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
    model.add(Dense(500,activation=('relu')))
    model.add(Dropout(.2))
    model.add(Dense(10,activation=('softmax'))) #This is the classification layer
    # print(model.summary())
    adam=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,loss=categorical_crossentropy,metrics=["accuracy"])

# Step 3: Create the ART classifier
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

# Step 4: Train the ART classifier
start_time=time.time()
classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs)#, verbose=False
end_time = time.time()
elapsed_time = end_time - start_time
print("{} - batch size : {} - epochs : {} - training in {:.2f} ms.".format(keras_file_name,batch_size,nb_epochs,elapsed_time*time_weight))

classifier.save(keras_file_name+'.h5', tmpdir)