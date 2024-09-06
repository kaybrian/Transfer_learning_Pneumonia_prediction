# -*- coding: utf-8 -*-
"""notebook3d46c8dfba

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/notebook3d46c8dfba-af9a7570-c09a-4a5f-8f9c-9044707fc10a.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240906/auto/storage/goog4_request%26X-Goog-Date%3D20240906T204030Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D730b4520f6ab4a704c54e9e6b806a61b0f130c16bb6b15e4a1a5adfda26197e8cf1eb2c65b3949291a418cd146251dc4c3da0b8fda1b9622d68f30f0bec8d749520e229966d97a76f644569fadebec0b2ff522bd22f09ac10df264d88315c29939b0643fe5dbcd39d83df8dcfeaf65e38dff96e58235dea45acd39d2950d86eb8f23ce45c8a1d7fcdc17e5f2d221fbc5d91048cea30a44a2cc8aac52dc8fa3a8235a4c6e7f1355904f784bb833557e63563144381442fc5d2a0fd05857d963bf852e7351f910d0d86ab7c4f385ee0dbd7b67b5db26ea24b0c7b9340e0dc20730edd41006a06ec2eb4e27ba27c19747c23e6e3c2c05e41df95eb154887d84657d
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'chest-xray-pneumonia:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F17810%2F23812%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240906%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240906T204029Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D323bc56283fe43c13030b2edfc16e1277a40940620462ca18168ce109b170fd2b63d6ba0533dac73ddd5a3d2483ebd8dc08ab76d0fccaf86715a4c33bcb3bd7b14f4a8b522f1889b557630bfc31bfbd97f4dfd04c32503a561e37a4e99c296cf20e08f72195d57b82ca7b0a9e346d75bad504a0930c5ad7c9d4843d62ee0ffb2770dc1fa5c574453dd2a59ed3f28db157eb3e1953fbf000115f5669e3bbedd759359322d3db4dc3e83d02160bcef329e3ce249ebde86aa1c379ff63bfb877ad8093985f50927a5e768931acb07f91e61b2217d4ff27b9fc23a97591bc08da306f6bbd912692fafdb949836494b2161022cccb12be84dc5d1bef39aa1f4d44b44'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install tensorflow numpy pandas matplotlib scikit-learn

# import the needed pacages for the training
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# paths to the data sets
train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'
val_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/val'
test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# scale the images
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from train images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# get the valdation images
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# get the testing images loaded in to the system'
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load VGG16 without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# layers for fine-tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

#  final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=2,
    verbose=1
)

"""**InceptionNet Implementation:**

- This function returns a Keras image classification model, optionally loaded with weights pre-trained on ImageNet.
"""

base_model_inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model_inceptionv3.layers:
    layer.trainable = False


# layers for fine-tuning
x = base_model_inceptionv3.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)


#  final model inception
model_inception = Model(inputs=base_model_inceptionv3.input, outputs=predictions)

# Compile the model
model_inception.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history_inception = model_inception.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=1
)

"""**EfficientNet Implementation:**

- This function returns a Keras image classification model, optionally loaded with weights pre-trained on ImageNet.

"""

# Load EfficientNet without the top layers
base_model_efficientNet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False


# layers for fine-tuning
x = base_model_efficientNet.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

#  final model
model_efficientNet = Model(inputs=base_model_efficientNet.input, outputs=predictions)

# Compile the model
model_efficientNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history_efficientNet = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=4,
    verbose=1
)

"""**Model Evaluation**

**Evaluate on Test Set:**

- The models will be evaluated in the follwoing order
 - Vgg16
 - InceptionNet
 - EfficientNet
"""

# Evaluating he Vgg16 Model first

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy VGG16 Mode:  {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Evaluating the InceptionNet
test_loss, test_accuracy = model_inception.evaluate(test_generator)
print(f"Test Accuracy InceptionNet Model: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Evaluating the EfficientNet
test_loss, test_accuracy = model_efficientNet.evaluate(test_generator)
print(f"Test Accuracy EfficientNet Model: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

"""**Detailed Performance Metrics**

"""

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

"""**VGG16 first**"""

# Predict on test data Vgg16
test_preds = model.predict(test_generator)
test_preds = (test_preds > 0.5).astype(int)

# Get true labels
true_labels = test_generator.classes

# Classification report
print(classification_report(true_labels, test_preds, target_names=['Normal', 'Pneumonia']))

"""**InceptionNet**"""

# Predict on test data InceptionNet

test_preds = model_inception.predict(test_generator)
test_preds = (test_preds > 0.5).astype(int)

# Get true labels
true_labels = test_generator.classes

# Classification report
print(classification_report(true_labels, test_preds, target_names=['Normal', 'Pneumonia']))

"""**EfficientNet**"""

# Predict on test data EfficientNet
test_preds = model_efficientNet.predict(test_generator)
test_preds = (test_preds > 0.5).astype(int)

# Get true labels
true_labels = test_generator.classes

# Classification report
print(classification_report(true_labels, test_preds, target_names=['Normal', 'Pneumonia']))

""" **Results Visualization**

 Here i will be able to visualize the Plotting Training History of each model starting with

 - Vgg16
 - InceptionNet
 - EfficientNet


 1. VGG16
"""

import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

"""2. InceptionNet"""

# Plot accuracy
plt.plot(history_inception.history['accuracy'], label='Train Accuracy')
plt.plot(history_inception.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot loss
plt.plot(history_inception.history['loss'], label='Train Loss')
plt.plot(history_inception.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

"""3 - EfficientNet"""

# Plot accuracy
plt.plot(history_efficientNet.history['accuracy'], label='Train Accuracy')
plt.plot(history_efficientNet.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot loss
plt.plot(history_efficientNet.history['loss'], label='Train Loss')
plt.plot(history_efficientNet.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')
plt.show()