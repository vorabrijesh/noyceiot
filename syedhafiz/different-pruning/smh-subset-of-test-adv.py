
import numpy as np
import sys
import argparse
import time

print('# '*50+str(time.ctime())+' :: smh-subset-of-test-adv')

per_class_samples = int(sys.argv[1])
n_classes = int(sys.argv[2])
dataset_name=str(sys.argv[3])
model_name=str(sys.argv[4])
attack_name=str(sys.argv[5])
n_test_adv_samples=str(sys.argv[6])
n_test_adv_samples_subset=n_classes*per_class_samples

x_test = np.load(dataset_name+'-x-test-'+str(n_test_adv_samples)+'.npy')
y_test = np.load(dataset_name+'-y-test-'+str(n_test_adv_samples)+'.npy')
x_test_adv = np.load(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-'+str(n_test_adv_samples)+'.npy')

print(x_test.shape)
print(y_test.shape)
print(x_test_adv.shape)
x_test_subset = []
y_test_subset = []
x_test_adv_subset = []
count = np.zeros(n_classes)
t=0
for y in y_test:
    pos = np.where(y==1)
    count[pos] =  count[pos]+1
    if count[pos]<=per_class_samples:
        x_test_subset.append(x_test[t])
        x_test_adv_subset.append(x_test_adv[t])
        y_test_subset.append(y)
        # print(str(pos)+'::'+str(count[pos]))
    t = t+1
x_test_subset = np.array(x_test_subset)
y_test_subset = np.array(y_test_subset)
x_test_adv_subset = np.array(x_test_adv_subset)
print(x_test_subset.shape)
print(y_test_subset.shape)
print(x_test_adv_subset.shape)
np.save(dataset_name+'-x-test-to-tensorrt-'+str(n_test_adv_samples_subset),x_test_subset)
np.save(dataset_name+'-y-test-to-tensorrt-'+str(n_test_adv_samples_subset),y_test_subset)
np.save(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-to-tensorrt-'+str(n_test_adv_samples_subset),x_test_adv_subset)