from art.attacks.attack import Attack
from art.attacks.evasion.auto_attack import AutoAttack
from art.attacks.evasion.carlini import CarliniL0Method, CarliniLInfMethod, CarliniL2Method
from art.attacks.evasion import DeepFool, FastGradientMethod, AutoProjectedGradientDescent, ShadowAttack, Wasserstein, BrendelBethgeAttack, ShapeShifter
from art.attacks.evasion.elastic_net import ElasticNet
from art.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np

from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, to_categorical
from art.utils import load_cifar10

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

ATTACK_NAME={"CW":"CarliniWagner", "DF":"Deepfool", "FGSM":"FastGradientMethod", "APGD":"AutoProjectedGradientDescent", "SA":"ShadowAttack", "WS":"Wasserstein",
            "EN":"ElasticNet","ADP":"AdversarialPatch"}
tmpdir = os.getcwd()

# # Step 2: Load Model
keras_file_name=str(sys.argv[1])
dataset_name=str(sys.argv[2])
model_name=str(sys.argv[3])
attack_name=str(sys.argv[4])
n_test_samples=int(sys.argv[5])

model = tf.keras.models.load_model(keras_file_name+'.h5')
x_test = np.load(dataset_name+'-x-test-'+str(n_test_samples)+'.npy')
y_test = np.load(dataset_name+'-y-test-'+str(n_test_samples)+'.npy')


classifier = KerasClassifier(model=model)#, clip_values=(min_pixel_value, max_pixel_value), use_logits=False
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {:.2f}%".format(accuracy * 100))


# Step 6: Generate adversarial test examples

if attack_name==ATTACK_NAME.get("CW"):
    attack = CarliniL2Method(classifier=classifier)
elif attack_name==ATTACK_NAME.get("DF"):
    attack = DeepFool(classifier=classifier, max_iter=5, batch_size=128, verbose=True)
elif attack_name==ATTACK_NAME.get("FGSM"):
    attack = FastGradientMethod(estimator=classifier, eps=0.2)
# elif attack_name==ATTACK_NAME.get("APGD"):
#     attack = AutoProjectedGradientDescent(estimator=classifier,eps=0.3,eps_step=0.1,max_iter=5,targeted=False,nb_random_init=1,batch_size=32,verbose=False)
# elif attack_name==ATTACK_NAME.get("SA"):
#     attack = ShadowAttack(estimator=classifier,sigma=0.5,nb_steps=3,learning_rate=0.1,lambda_tv=0.3,lambda_c=1.0,lambda_s=0.5,batch_size=32,targeted=False,verbose=False)
elif attack_name==ATTACK_NAME.get("EN"):
    attack = ElasticNet(classifier=classifier,targeted=False, max_iter=2, verbose=True)
elif attack_name==ATTACK_NAME.get("ADP"):
    attack = AdversarialPatch(classifier=classifier,rotation_max=0.5,scale_min=0.4,scale_max=0.41,learning_rate=5.0,batch_size=10,max_iter=5,verbose=True)
elif attack_name==ATTACK_NAME.get("WS"):
    attack = Wasserstein(classifier,regularization=100,conjugate_sinkhorn_max_iter=5,
            projected_sinkhorn_max_iter=5,norm="wasserstein",ball="wasserstein",targeted=False,p=2,eps_iter=2,eps_factor=1.05,eps_step=0.1,kernel_size=5,batch_size=5,verbose=True)

start_time=time.time()
x_test_adv = attack.generate(x_test)
end_time = time.time()
elapsed_time = end_time - start_time

# Step 7: Evaluate the ART classifier on adversarial test examples

np.save(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-'+str(n_test_samples),x_test_adv)
# np.save('cifar10'+'-Deepfool-'+'x-test-adv',x_test_adv)
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {:.2f}% - in {:.2f} ms.".format(accuracy * 100,elapsed_time*1000))

# f.close()
# sys.stdout = original_stdout