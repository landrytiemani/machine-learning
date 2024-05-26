# Purpose: This script trains a Keras convolutional neural network (CNN) model for image classification.
# It loads image data from directories, performs data augmentation, defines the model architecture, and compiles the model for training.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Image parameters
image_list = ["0", "1"]
output_n = len(image_list)  # Number of output classes
img_width, img_height = 50, 50  # Target image size
channels = 3  # RGB image channels

# Paths to image folders
path_train = "data/images/train"
path_valid = "data/images/valid"

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zca_whitening=True,
    validation_split=0.2
)

# Rescaling for validation data (no augmentation)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# Generate training image data with augmentation
train_generator = train_datagen.flow_from_directory(
    path_train,
    target_size=(img_width, img_height),
    color_mode="rgb",
    class_mode='categorical',
    batch_size=32,  # Adjust batch size as needed
    shuffle=True,
    classes=image_list,
    subset='training',
    seed=42
)

# Generate validation image data
valid_generator = valid_datagen.flow_from_directory(
    path_valid,
    target_size=(img_width, img_height),
    color_mode="rgb",
    class_mode='categorical',
    batch_size=32,  # Adjust batch size as needed
    shuffle=True,
    classes=image_list,
    subset='validation',
    seed=42
)

# Calculate number of training and validation samples (not in original code but can be useful)
train_samples = train_generator.samples
valid_samples = valid_generator.samples

print("Number of training samples:", train_samples)
print("Number of validation samples:", valid_samples)