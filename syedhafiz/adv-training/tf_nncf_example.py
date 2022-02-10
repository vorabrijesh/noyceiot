import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization,Activation,Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD

import sys
import numpy as np
from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model, register_default_init_args
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50

# script.py <path_to_json>

json_path = sys.argv[1]

nncf_config = NNCFConfig.from_json(json_path)

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
print("Train image object: {}, Train label object: {}, Test image object: {}, Test label object: {}".format(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape))
batch_size = 128
epochs=2
per_class_samples = 50
n_classes = 10
x_train_subset = []
y_train_subset = []
count = np.zeros(n_classes)
t=0
for y in train_labels:
    count[y] =  count[y]+1
    if count[y]<=per_class_samples:
        x_train_subset.append(train_images[t])
        y_train_subset.append(y)
    t = t+1
x_train_subset = np.array(x_train_subset)
y_train_subset = np.array(y_train_subset)
y_train_subset = to_categorical(y_train_subset)
print("Train image object: {}, Train label object: {}, Test image object: {}, Test label object: {}".format(x_train_subset.shape, y_train_subset.shape, test_images.shape, test_labels.shape))

train_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(x_train_subset, tf.float32),tf.cast(y_train_subset, tf.int64)))
train_dataset = train_dataset.shuffle(n_classes*per_class_samples).batch(batch_size)

# print(y_train_subset[:200])
# model = ResNet50(include_top=False, weights='imagenet',input_shape=(32,32,3), classes=train_labels.shape[1])
#representative_dataset = tf.data.Dataset.list_files("/path/*.jpeg")
model = tf.keras.models.load_model('classifier-ResNet50-cifar10-on-500.h5')

# model= Sequential()
# base_model = ResNet50(include_top=False,weights='imagenet',input_shape=(32,32,3))
# model.add(base_model)
# model.add(Flatten())
# model.add(Dense(4000,activation=('relu'),input_dim=512))
# model.add(Dense(2000,activation=('relu'))) 
# model.add(Dropout(.4))
# model.add(Dense(1000,activation=('relu'))) 
# model.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
# model.add(Dense(500,activation=('relu')))
# model.add(Dropout(.2))
# model.add(Dense(10,activation=('softmax'))) #This is the classification layer
# print(model.summary())
# backbone = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
# output = tf.keras.layers.Dense(n_classes, activation='softmax', name='predictions')(backbone.output)
# model = tf.keras.Model(backbone.input, output)
# # #adam=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(optimizer= 'adam',loss=categorical_crossentropy,metrics=["accuracy"])
# model.fit(x=x_train_subset, y=y_train_subset, batch_size=batch_size, epochs=epochs, verbose=1)
nncf_config = register_default_init_args(nncf_config, train_dataset, batch_size=batch_size)

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