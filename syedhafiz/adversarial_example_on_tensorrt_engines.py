import os
import time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import convert_to_constants
import sys

input_1d = 224
tmpdir = os.getcwd()
original_stdout = sys.stdout 
f = open("resultmobilenet_adversary_examples.txt", "a")
sys.stdout = f

print("\n\n********* New run ************\n")
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)
# YellowLabradorLooking_new
# file = tf.keras.utils.get_file("YellowLabradorLooking_new.jpg", "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg")
# file = tf.keras.utils.get_file("grace_hopper.jpg","https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
img = tf.keras.preprocessing.image.load_img('epsilon_dot_25.jpg', target_size=[input_1d, input_1d])
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])

pretrained_model = tf.keras.applications.MobileNet(weights="imagenet")
start_time = time.time()
image_probs = pretrained_model.predict(x)
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_milliSeconds = elapsed_time*1000
print("Regular inference in milliseconds: ",elapsed_time_milliSeconds)

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

_, image_class, class_confidence = get_imagenet_label(image_probs)
print('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))

mobilenet_save_path = os.path.join(tmpdir, "mobilenet/1/")
tf.saved_model.save(pretrained_model, mobilenet_save_path)

def my_input_fn():
  size = [input_1d, input_1d]
  inp1 = np.random.normal(size=(1, *size, 3)).astype(np.float32)
  yield [inp1]

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<25))
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(maximum_cached_engines=100)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=mobilenet_save_path, conversion_params=conversion_params)
converter.convert()

converter.build(input_fn=my_input_fn)
output_saved_model_dir = mobilenet_save_path
converter.save(output_saved_model_dir)

saved_model_loaded = tf.saved_model.load(output_saved_model_dir, tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
input_tensor=tf.constant(x)

start_time = time.time()
output_tensor = frozen_func(input_tensor)
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_milliSeconds = elapsed_time*1000
print("Optimized inference in milliseconds: ",elapsed_time_milliSeconds)

_, trt_image_class, trt_class_confidence = get_imagenet_label(output_tensor[0].numpy())
print('{} : {:.2f}% Confidence'.format(trt_image_class, trt_class_confidence*100))

f.close()
sys.stdout = original_stdout
