import matplotlib as mpl
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img

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

time_weight = 1000
n_trials = 2
epsilons = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
input_1d = 224
n_classes = 1000
num_images_validation_per_class = 10
n_images = n_classes*num_images_validation_per_class
topk=5
validation_file = open("ILSVRC2012_validation_ground_truth_v2.txt", "r")
ground_truth_map = []
line = validation_file.readline()
while line:
	ground_truth_map.append(int(line))
	line = validation_file.readline()
validation_file.close()
tmpdir = os.getcwd()
original_stdout = sys.stdout 
f = open("joined_attack_val_3_top5_int8_trial_2.txt", "w")
sys.stdout = f

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
loss_object = tf.keras.losses.CategoricalCrossentropy()

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

def my_input_fn():
  size = [input_1d, input_1d]
  inp1 = np.random.normal(size=(1, *size, 3)).astype(np.float32)
  yield [inp1]

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)
  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

# print("\n********* New run ************\n")

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)

pretrained_model = tf.keras.applications.MobileNetV2(weights='imagenet')

mobilenet_v2_save_path = os.path.join(tmpdir, 'mobilenet_v2_int8/')
tf.saved_model.save(pretrained_model, mobilenet_v2_save_path)

params = tf.experimental.tensorrt.ConversionParams(precision_mode='INT8', maximum_cached_engines=1, use_calibration=True)
converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=mobilenet_v2_save_path, conversion_params=params)

def my_calibration_input_fn():
  size = [input_1d, input_1d]
  inp1 = np.random.normal(size=(1, *size, 3)).astype(np.float32)
  yield [inp1]

converter.convert(calibration_input_fn=my_calibration_input_fn)

converter.build(input_fn=my_input_fn)

output_saved_model_dir = mobilenet_v2_save_path
converter.save(output_saved_model_dir)

# conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
# conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
# conversion_params = conversion_params._replace(precision_mode="FP16")
# conversion_params = conversion_params._replace(maximum_cached_engines=100)

# converter = trt.TrtGraphConverterV2(input_saved_model_dir=mobilenet_v2_save_path, conversion_params=conversion_params)
# converter.convert()

# converter.build(input_fn=my_input_fn)
# output_saved_model_dir = mobilenet_v2_save_path
# converter.save(output_saved_model_dir)

saved_model_loaded = tf.saved_model.load(output_saved_model_dir, tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)


for dir in range(1,n_classes):
	list_files = listdir('../validation3/'+str(dir)+'/')
	for im_no in range(num_images_validation_per_class):
		im_file = list_files[im_no]
		image_path = '../validation3/'+str(dir)+'/'+im_file
		img = tf.keras.preprocessing.image.load_img(image_path, target_size=[input_1d, input_1d])
		x = tf.keras.preprocessing.image.img_to_array(img)
		image = tf.keras.applications.mobilenet_v2.preprocess_input(x[tf.newaxis,...])

		image_no_in_validation_dir = int(im_file[18:23])
		cur_label = ground_truth_map[image_no_in_validation_dir-1]
		print('***********'+str(image_no_in_validation_dir)+' :: '+str(cur_label))

		image_probs = pretrained_model.predict(image)

		# Get the input label of the image.
		output_tensor_to_array = image_probs[0]
		label_reengineered = np.where(output_tensor_to_array==np.max(output_tensor_to_array))
		label = tf.one_hot(label_reengineered, image_probs.shape[-1])
		label = tf.reshape(label, (1, image_probs.shape[-1]))
		print('GTruth:: PL: {}, VL: {}'.format(label_reengineered, cur_label))
		image_for_adv = tf.constant(image)
		perturbations = create_adversarial_pattern(image_for_adv, label)

		for i, eps in enumerate(epsilons):
			print('\n*********'+str(i)+' Epsilon: '+str(eps)+' ************\n')
			adv_x = image + eps*perturbations
			adv_x = tf.clip_by_value(adv_x, -1, 1)

			inference_time_mean = []
			confidence_mean = []
			image_probs = pretrained_model.predict(adv_x)
			regular_output_tensor_to_array = image_probs[0]
			regular_top_values_index = sorted(range(len(regular_output_tensor_to_array)), key=lambda i: regular_output_tensor_to_array[i])[-topk:]
			# regular_label_reengineered = np.where(regular_output_tensor_to_array==np.max(regular_output_tensor_to_array))
			print('RL: {}'.format(regular_top_values_index))
			for trial in range(1,n_trials):
				start_time = time.time()
				image_probs = pretrained_model.predict(adv_x)
				end_time = time.time()
				elapsed_time = end_time - start_time
				inference_time_mean.append(elapsed_time*time_weight)
				_, image_class, class_confidence = get_imagenet_label(image_probs)
				
				confidence_mean.append(class_confidence*100)
			print('RI {:.2f} +- {:.2f} ms'.format(np.mean(inference_time_mean), np.std(inference_time_mean)))
			print('{} : {:.2f} +- {:.2f} % Confidence'.format(image_class, np.mean(confidence_mean), np.std(confidence_mean)))

			optimized_inference_time_mean = []
			optimized_confidence_mean = []
			input_tensor=tf.constant(adv_x)
			output_tensor = frozen_func(input_tensor)
			optimized_output_tensor_to_array = output_tensor[0].numpy()[0]
			# optimized_label_reengineered = np.where(optimized_output_tensor_to_array==np.max(optimized_output_tensor_to_array))
			optimized_top_values_index = sorted(range(len(optimized_output_tensor_to_array)), key=lambda i: optimized_output_tensor_to_array[i])[-topk:]
			print('OL: {}'.format(optimized_top_values_index))
			for trial in range(1,n_trials):
				start_time = time.time()
				output_tensor = frozen_func(input_tensor)
				end_time = time.time()
				elapsed_time = end_time - start_time
				optimized_inference_time_mean.append(elapsed_time*time_weight)
				_, trt_image_class, trt_class_confidence = get_imagenet_label(output_tensor[0].numpy())
				optimized_confidence_mean.append(trt_class_confidence*100)
			print('OI {:.2f} +- {:.2f} ms'.format(np.mean(optimized_inference_time_mean), np.std(optimized_inference_time_mean)))
			print('{} : {:.2f} +- {:.2f} % Confidence'.format(trt_image_class, np.mean(optimized_confidence_mean), np.std(optimized_confidence_mean)))	
f.close()
sys.stdout = original_stdout
