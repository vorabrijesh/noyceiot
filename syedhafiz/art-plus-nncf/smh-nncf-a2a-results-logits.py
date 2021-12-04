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
# import tensorflow_datasets as tfds
from tensorflow.python.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization,Activation,Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD

import sys
import numpy as np
# from nncf import NNCFConfig
# from nncf.tensorflow import create_compressed_model, register_default_init_args
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from art.estimators.classification import KerasClassifier
from art.attacks.attack import Attack
from art.attacks.evasion.auto_attack import AutoAttack
from art.attacks.evasion.carlini import CarliniL0Method, CarliniLInfMethod, CarliniL2Method
from art.attacks.evasion import DeepFool, FastGradientMethod, AutoProjectedGradientDescent, ShadowAttack, Wasserstein, BrendelBethgeAttack, ShapeShifter, UniversalPerturbation, NewtonFool
from art.attacks.evasion.iterative_method import BasicIterativeMethod
from art.attacks.evasion.elastic_net import ElasticNet
from art.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
tf.compat.v1.disable_eager_execution()
# original_stdout = sys.stdout 
# f = open("output-tensorrt-results.txt", "a")
# sys.stdout = f
print('# '*50+str(time.ctime())+' :: nncf-a2a-results')
ATTACK_NAME={"APGD":"AutoProjectedGradientDescent", "WS":"Wasserstein"}

time_weight=1000
tmpdir = os.getcwd()
dataset_name=str(sys.argv[1])
model_name=str(sys.argv[2])
attack_name=str(sys.argv[3])
n_test_adv_samples_subset=int(sys.argv[4])
keras_file_name=str(sys.argv[5])
json_path = str(sys.argv[6])
batch_size = int(sys.argv[7])

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)
dataset_name+'-x-test-to-tensorrt-'+str(n_test_adv_samples_subset)
x_test = np.load(dataset_name+'-x-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
y_test = np.load(dataset_name+'-y-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')

input_tensor=tf.constant(x_test.astype('float32'))
optimization_str = json_path.split('/')[-1].split('.')[0]
compressed_model=tf.keras.models.load_model(optimization_str+'-'+keras_file_name+"_logits.h5")
classifier = KerasClassifier(model=compressed_model,clip_values=(0, 1), use_logits=True)#, clip_values=(min_pixel_value, max_pixel_value), use_logits=False
classifier.predict(x_test)
start_time=time.time()
predictions = classifier.predict(x_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("NNCF stats on benign test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
predictions = tf.round(tf.nn.softmax(predictions))
predictions = predictions.eval(session=tf.compat.v1.Session())
process_results(predictions, y_test)

if attack_name==ATTACK_NAME.get("APGD"):
    attack = AutoProjectedGradientDescent(estimator=classifier,eps=0.3,eps_step=0.1,max_iter=5,targeted=False,nb_random_init=1,batch_size=batch_size,verbose=True)
elif attack_name==ATTACK_NAME.get("WS"):
    attack = Wasserstein(classifier,regularization=100,conjugate_sinkhorn_max_iter=5, projected_sinkhorn_max_iter=5,norm="wasserstein",ball="wasserstein",targeted=False,p=2,eps_iter=2,eps_factor=1.05,eps_step=0.1,kernel_size=5,batch_size=batch_size,verbose=True)

start_time=time.time()
x_test_adv = attack.generate(x_test)
end_time = time.time()
elapsed_time = end_time - start_time
np.save(dataset_name+'-'+optimization_str+"-"+model_name+'-'+attack_name+'-x-test-adv-'+str(n_test_adv_samples_subset),x_test_adv)

print(attack_name+":: Adversarial examples on NNCF-optimized model generation time: {:.2f} ms.".format(elapsed_time*time_weight))

input_tensor=tf.constant(x_test_adv.astype('float32'))
start_time=time.time()
predictions = classifier.predict(x_test_adv)
end_time = time.time()
elapsed_time = end_time - start_time
print("NNCF stats on adversarial test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
predictions = tf.round(tf.nn.softmax(predictions))
predictions = predictions.eval(session=tf.compat.v1.Session())
process_results(predictions, y_test)

full_bone_model = tf.keras.models.load_model(keras_file_name+'_logits.h5')
full_bone_classifier = KerasClassifier(model=full_bone_model,clip_values=(0, 1))#, clip_values=(min_pixel_value, max_pixel_value), use_logits=False
full_bone_classifier.predict(x_test_adv)
start_time=time.time()
predictions = full_bone_classifier.predict(x_test_adv)
end_time = time.time()
elapsed_time = end_time - start_time
print("Full-bone stats on adversarial test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
predictions = tf.round(tf.nn.softmax(predictions))
predictions = predictions.eval(session=tf.compat.v1.Session())
process_results(predictions, y_test)