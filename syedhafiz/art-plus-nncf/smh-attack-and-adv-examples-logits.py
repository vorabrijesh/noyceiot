from art.attacks.evasion import AutoProjectedGradientDescent, AutoAttack, Wasserstein
from smh_utility_process_results import process_results
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import numpy as np

from art.estimators.classification import KerasClassifier
from tensorflow.keras import backend as K
import time

import warnings
#suppress warnings
warnings.filterwarnings('ignore')

###############
import matplotlib as mpl
import os
from os import listdir
import time
# import tensorflow as tf
# import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import convert_to_constants
import sys


print('# '*50+str(time.ctime())+' :: smh-attack-and-adv-examples')

ATTACK_NAME={"AA":"AutoAttack", "APGD":"AutoProjectedGradientDescent", "WS":"Wasserstein",}
tmpdir = os.getcwd()
time_weight=1000
# # Step 2: Load Model
keras_file_name=str(sys.argv[1])
dataset_name=str(sys.argv[2])
model_name=str(sys.argv[3])
attack_name=str(sys.argv[4])
n_test_samples=int(sys.argv[5])
K.clear_session()
model = tf.keras.models.load_model(keras_file_name+'.h5')
x_test = np.load(dataset_name+'-x-test-'+str(n_test_samples)+'.npy')
y_test = np.load(dataset_name+'-y-test-'+str(n_test_samples)+'.npy')


classifier = KerasClassifier(model=model,clip_values=(0, 1), use_logits=True)#, clip_values=(min_pixel_value, max_pixel_value), use_logits=False
# classifier.predict(x_test)
start_time=time.time()
predictions = classifier.predict(x_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("Full-bone stats on benign test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
predictions = tf.round(tf.nn.softmax(predictions))
predictions = predictions.eval(session=tf.compat.v1.Session())
process_results(predictions, y_test)


# Step 6: Generate adversarial test examples
flag_TUP_Attack=False
if attack_name==ATTACK_NAME.get("APGD"):
    attack = AutoProjectedGradientDescent(estimator=classifier,eps=0.3,eps_step=0.1,max_iter=5,targeted=False,nb_random_init=1,batch_size=128,verbose=True)
# elif attack_name==ATTACK_NAME.get("AA"):
    # attack = AutoAttack(estimator=classifier, norm=np.inf, eps=0.3, eps_step=0.1, attacks=None, batch_size=128, estimator_orig=None)
elif attack_name==ATTACK_NAME.get("WS"):
    attack = Wasserstein(classifier,regularization=100,conjugate_sinkhorn_max_iter=5, projected_sinkhorn_max_iter=5,norm="wasserstein",ball="wasserstein",targeted=False,p=2,eps_iter=2,eps_factor=1.05,eps_step=0.1,kernel_size=5,batch_size=128,verbose=True)

start_time=time.time()
x_test_adv = attack.generate(x_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("Adversarial examples generation time: {:.2f} ms.".format(elapsed_time*time_weight))
# Step 7: Evaluate the ART classifier on adversarial test examples

np.save(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-'+str(n_test_samples),x_test_adv)
# classifier.predict(x_test_adv)
start_time = time.time()
predictions = classifier.predict(x_test_adv)
end_time = time.time()
elapsed_time = end_time - start_time
print("Full-bone stats on adversarial test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
predictions = tf.round(tf.nn.softmax(predictions))
predictions = predictions.eval(session=tf.compat.v1.Session())
process_results(predictions, y_test)