import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.datasets import cifar10
import sys
import numpy as np
from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model, register_default_init_args
from sklearn.metrics import classification_report

# script.py <path_to_json>

json_path = sys.argv[1]

# Instantiate your uncompressed model
from tensorflow.keras.applications import ResNet50
#model = ResNet50(include_top=False,weights='imagenet',input_shape=(32,32,3),classes=y_train.shape[1])

# Load a configuration file to specify compression
nncf_config = NNCFConfig.from_json(json_path)

# Provide dataset for compression algorithm initialization
#train_ds, test_ds = tfds.load('cifar10', split=['train','test'], shuffle_files=True)
# Load CIFAR 10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

print(test_labels.shape)
print(test_labels[0])

batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(train_images, tf.float32),tf.cast(train_labels, tf.int64)))
train_dataset = train_dataset.shuffle(50000).batch(batch_size)

print(train_labels.shape[1])
model = ResNet50(include_top=False, weights='imagenet',input_shape=(32,32,3), classes=train_labels.shape[1])
#representative_dataset = tf.data.Dataset.list_files("/path/*.jpeg")
nncf_config = register_default_init_args(nncf_config, train_dataset, batch_size=1)

# Apply the specified compression algorithms to the model
print("Compression Algorithm...")
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
print("Model Compression Complete")

# Save compressed model
#compressed_model.save('resnet50_cifar_quantization.h5')
# Now use compressed_model as a usual Keras model
# to fine-tune compression parameters along with the model weights
print("Predicting...")
predictions = compressed_model.predict(test_images)
#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(test_labels, axis=1)) / len(test_labels)
#print("Accuracy on benign test examples: {:.2f}%".format(accuracy * 100))
predictions_classes = np.argmax(predictions, axis=1)
#true_classes = np.argmax(test_labels,axis=1)
print(classification_report(test_labels, predictions_classes))
# ... the rest of the usual TensorFlow-powered training pipeline

# Export to Frozen Graph, TensorFlow SavedModel or .h5  when done fine-tuning 
#compressed_model.save('resnet50_cifar_quantization.h5')
#compression_ctrl.export_model("compressed_model.pb", save_format='frozen_graph')