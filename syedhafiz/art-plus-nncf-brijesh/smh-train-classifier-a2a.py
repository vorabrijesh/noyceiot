from art.attacks.evasion.carlini import CarliniL0Method, CarliniLInfMethod
import tensorflow as tf
from tensorflow.python.keras.activations import get
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.backend import flatten
from sklearn import preprocessing
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization,Activation,Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import ResNet50, VGG19, MobileNet, DenseNet121
from tensorflow.python.keras.datasets import cifar10, cifar100
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

print('# '*50+str(time.ctime()))
tmpdir = os.getcwd()
time_weight=5000
MODEL_NAME={"VGG":"VGG19", "RESNET":"ResNet50", "MOBILENET": "MobileNet", "DENSENET":"DenseNet121"}

n_train_samples = int(sys.argv[1])
batch_size = int(sys.argv[2])
nb_epochs= int(sys.argv[3])
keras_file_name=str(sys.argv[4])
model_name=str(sys.argv[5])
dataset_name = str(sys.argv[6])

if dataset_name=='cifar10':
    (x_train, y_train), _, min_pixel_value, max_pixel_value = load_cifar10()
elif dataset_name=='cifar100':
    (x_train, y_train),( _,_)  = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    print("xtrain ", x_train.shape)
    print("ytrain ", y_train.shape)
    y_train = (preprocessing.LabelBinarizer().fit_transform(y_train))
    y_train = y_train.astype('float64')
    x_train = x_train.astype('float64')
    min_pixel_value = 0.0
    max_pixel_value = 1.0


x_train = x_train[:n_train_samples]
y_train = y_train[:n_train_samples]
print((x_train.shape,y_train.shape))

# # model= Sequential()
# # Step 2: Load Model and do transfer learning 
if model_name==MODEL_NAME.get("VGG"):
    backbone = VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name==MODEL_NAME.get("RESNET"):
    backbone = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("MOBILENET"):
    backbone = MobileNet(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("DENSENET"):
    backbone = DenseNet121(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')

output = tf.keras.layers.Dense(y_train.shape[1], activation='softmax', name='predictions')(backbone.output)
model = tf.keras.Model(backbone.input, output)
model.compile(optimizer= 'adam',loss=categorical_crossentropy,metrics=["accuracy"])
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

start_time=time.time()
classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs)#, verbose=False
end_time = time.time()
elapsed_time = end_time - start_time
print("{} - batch size : {} - epochs : {} - training in {:.2f} ms.".format(keras_file_name,batch_size,nb_epochs,elapsed_time*time_weight))
classifier.save(keras_file_name+'.h5', tmpdir)