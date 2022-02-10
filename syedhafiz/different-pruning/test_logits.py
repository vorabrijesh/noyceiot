import tensorflow as tf
import sys

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization,Activation,Dropout, Conv2D, MaxPooling2D

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import MobileNet

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import AutoProjectedGradientDescent, AutoAttack, Wasserstein
from art.utils import load_cifar10

import numpy as np
from sklearn.metrics import classification_report
to_percentage=100

def process_results(predictions, ground_true):
    predictions_classes = np.argmax(predictions, axis=1)
    print("Prediction: {}".format(predictions_classes))
    true_classes = np.argmax(ground_true,axis=1)
    # print('{},{}'.format(predictions_classes.shape,predictions_classes[0]))
    # print('{},{}'.format(true_classes.shape,true_classes[0]))
    res_dic = classification_report(true_classes, predictions_classes, output_dict=True)
    acc = res_dic.get("accuracy")
    mac_precision = res_dic.get("macro avg").get("precision")
    mac_recall = res_dic.get("macro avg").get("recall")
    mac_f1_score = res_dic.get("macro avg").get("f1-score")
    support = res_dic.get("macro avg").get("support")
    print('accuracy : {:.4f}; precision : {:.4f}; recall : {:.4f}; f1-score : {:.4f}; support : {} '.format(acc, mac_precision, mac_recall, mac_f1_score, support))#mac_precision)

n_train_samples = 500 #int(sys.argv[1])
batch_size = 128 #int(sys.argv[2])
nb_epochs= 2 #int(sys.argv[3])
# keras_file_name= str(sys.argv[4])
# model_name=str(sys.argv[5])


(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

x_train = x_train[:n_train_samples]
y_train = y_train[:n_train_samples]
x_test = x_test[:100]
y_test = y_test[:100]
print((x_train.shape,y_train.shape))

# https://www.tensorflow.org/guide/keras/train_and_evaluate
backbone = MobileNet(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
output = tf.keras.layers.Dense(y_train.shape[1], name='predictions')(backbone.output)
model = tf.keras.Model(backbone.input, output)
model.compile(optimizer= 'adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=True)
classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs)

predictions = classifier.predict(x_test)
predictions = tf.round(tf.nn.softmax(predictions))
predictions = predictions.eval(session=tf.compat.v1.Session())
print("Results for benign examples")
process_results(predictions, y_test)

print("Feeding to APGD...")
attack = AutoProjectedGradientDescent(estimator=classifier,eps=0.3,eps_step=0.1,max_iter=5,targeted=False,nb_random_init=1,batch_size=32,verbose=False)
print("Success!")

x_test_adv = attack.generate(x_test)

predictions = classifier.predict(x_test_adv)
predictions = tf.round(tf.nn.softmax(predictions))
predictions = predictions.eval(session=tf.compat.v1.Session())
print("Results for APGD generated adversarial examples")
process_results(predictions, y_test)

# TODO: Need to fix for AutoAttack
# print("Feeding to AutoAttack...")
# attack = AutoAttack(estimator=classifier, norm=np.inf, eps=0.3, eps_step=0.1, attacks=None, batch_size=32, estimator_orig=None)
# print("Success!")

print("Feeding to Wasserstein...")
attack = Wasserstein(classifier,regularization=100,conjugate_sinkhorn_max_iter=5, projected_sinkhorn_max_iter=5,norm="wasserstein",ball="wasserstein",targeted=False,p=2,eps_iter=2,eps_factor=1.05,eps_step=0.1,kernel_size=5,batch_size=5,verbose=True)
print("Success")

