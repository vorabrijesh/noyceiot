import matplotlib as mpl
import os
from os import listdir
import time
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
from sklearn.metrics import classification_report
from smh_utility_process_results import process_results
# original_stdout = sys.stdout 
# f = open("output-h5-to-tensorrt-fp16.txt", "a")
# sys.stdout = f
print('# '*50+str(time.ctime())+' :: keras-to-tensorrt')

time_weight=1000
input_1d = int(sys.argv[1])
dataset_name=str(sys.argv[2])
model_name=str(sys.argv[3])
attack_name=str(sys.argv[4])
n_test_adv_samples_subset=int(sys.argv[5])
keras_file_name=str(sys.argv[6])
tmpdir = os.getcwd()
def my_input_fn():
  size = [input_1d, input_1d]
  inp1 = np.random.normal(size=(1, *size, 3)).astype(np.float32)
  print("input_tensor: {}".format(inp1.shape))
  yield [inp1]

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)

pretrained_model = tf.keras.models.load_model(keras_file_name+'.h5')#return type is sequential
x_test = np.load(dataset_name+'-x-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
y_test = np.load(dataset_name+'-y-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
x_test_adv = np.load(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')

predictions = pretrained_model.predict(x_test)
start_time = time.time()
predictions = pretrained_model.predict(x_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("Full-bone stats on benign test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
process_results(predictions, y_test)

predictions = pretrained_model.predict(x_test_adv)
start_time = time.time()
predictions = pretrained_model.predict(x_test_adv)
end_time = time.time()
elapsed_time = end_time - start_time
print("Full-bone stats on adversarial test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
process_results(predictions, y_test)

tf_save_path = os.path.join(tmpdir, dataset_name+'-'+model_name+'-'+attack_name+'-tf/')
tf.saved_model.save(pretrained_model, tf_save_path)

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(maximum_cached_engines=100)#100

converter = trt.TrtGraphConverterV2(input_saved_model_dir=tf_save_path, conversion_params=conversion_params)
converter.convert()
converter.build(input_fn=my_input_fn)
output_saved_model_dir = tf_save_path
converter.save(output_saved_model_dir)

# f.close()
# sys.stdout = original_stdout