import matplotlib as mpl
import os
from os import listdir
import time
from numpy.core.fromnumeric import shape
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import convert_to_constants
import sys
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from smh_utility_process_results import process_results
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization,Activation,Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD

import sys
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50

# original_stdout = sys.stdout 
# f = open("output-tensorrt-results.txt", "a")
# sys.stdout = f
print('# '*50+str(time.ctime())+' :: tensorrt-results')


keras_file_name=str(sys.argv[1])

pretrained_model = tf.keras.models.load_model(keras_file_name+'.h5')
pretrained_model.summary()
print(pretrained_model.count_params())
