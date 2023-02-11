import numpy as np
import sys
from art.utils import load_cifar10
import argparse
import time
import tensorflow as tf
from sklearn import preprocessing

print('# '*50+str(time.ctime())+' :: smh-subset-of-test')

per_class_samples = int(sys.argv[1])
n_classes = int(sys.argv[2])
dataset_name=str(sys.argv[3])
if dataset_name=='cifar10':
    _, (x_test, y_test), _, _ = load_cifar10()
elif dataset_name=='cifar100':
    (_, _),(x_test,y_test)  = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    y_test = (preprocessing.LabelBinarizer().fit_transform(y_test))
    y_test = y_test.astype('float64')
    x_test = x_test.astype('float64')
print(x_test.shape)
print(y_test.shape)
x_test_subset = []
y_test_subset = []
count = np.zeros(n_classes)
t=0
for y in y_test:
    pos = np.where(y==1)
    count[pos] =  count[pos]+1
    if count[pos]<=per_class_samples:
        x_test_subset.append(x_test[t])
        y_test_subset.append(y)
        # print(str(pos)+'::'+str(count[pos]))
    t = t+1
x_test_subset = np.array(x_test_subset)
y_test_subset = np.array(y_test_subset)
print(x_test_subset.shape)
print(y_test_subset.shape)
n_test_samples=n_classes*per_class_samples
np.save(dataset_name+'-x-test-'+str(n_test_samples), x_test_subset)
np.save(dataset_name+'-y-test-'+str(n_test_samples), y_test_subset)