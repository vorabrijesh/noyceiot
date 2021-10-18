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

# original_stdout = sys.stdout 
# f = open("output-tensorrt-results.txt", "a")
# sys.stdout = f
print('# '*50+str(time.ctime())+' :: tensorrt-results')

time_weight=1000
tmpdir = os.getcwd()
dataset_name=str(sys.argv[1])
model_name=str(sys.argv[2])
attack_name=str(sys.argv[3])
n_test_adv_samples_subset=int(sys.argv[4])
# keras_file_name=str(sys.argv[6])

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)
dataset_name+'-x-test-to-tensorrt-'+str(n_test_adv_samples_subset)
x_test = np.load(dataset_name+'-x-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
y_test = np.load(dataset_name+'-y-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
x_test_adv = np.load(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
                    #  dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-to-tensorrt-'+str(n_test_adv_samples_subset)
tf_save_path = os.path.join(tmpdir, dataset_name+'-'+model_name+'-'+attack_name+'-tf/')

saved_model_loaded = tf.saved_model.load(tf_save_path, tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)

input_tensor=tf.constant(x_test.astype('float32'))
# print("xtest: {}, xtestasfp32: {}, inputtensor: {} ytest: {}, xtestadv: {}".format(x_test.shape,x_test.astype('float32').shape,input_tensor.shape,y_test.shape,x_test_adv.shape))
output_tensor = frozen_func(input_tensor)
start_time=time.time()
output_tensor = frozen_func(input_tensor)
end_time = time.time()
elapsed_time = end_time - start_time
optimized_output_tensor_to_array = np.array(output_tensor)[0,:,:]#output_tensor[0].numpy()[1]
print("TensorRT stats on benign test examples with inference in {:.2f} ms.".format( elapsed_time*time_weight))
process_results(optimized_output_tensor_to_array, y_test)

input_tensor=tf.constant(x_test_adv)
output_tensor = frozen_func(input_tensor)
start_time=time.time()
output_tensor = frozen_func(input_tensor)
end_time = time.time()
elapsed_time = end_time - start_time

optimized_output_tensor_to_array = np.array(output_tensor)[0,:,:]#output_tensor[0].numpy()[1]
accuracy = np.sum(np.argmax(optimized_output_tensor_to_array, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("TensorRT stats on adversarial test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
process_results(optimized_output_tensor_to_array, y_test)
# f.close()
# sys.stdout = original_stdout