# Title: Deep Learning Image Classification in Python using Keras

# Purpose: This script demonstrates how to build, train, and evaluate a convolutional neural network (CNN) model for image classification using Keras and TensorFlow. It includes data loading, preprocessing, model architecture, training, evaluation, and saving the model.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

import time
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Set the image parameters
image_list = ["0", "1"]
output_n = len(image_list)
img_width, img_height = 50, 50
channels = 3
batch_size = 2
epochs = 10

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

# Rescaling for validation and test data (no augmentation)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# Create image generators
train_generator = train_datagen.flow_from_directory(
    path_train,
    target_size=(img_width, img_height),
    color_mode="rgb",
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    classes=image_list,
    subset='training',
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    path_valid,
    target_size=(img_width, img_height),
    color_mode="rgb",
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    classes=image_list,
    subset='validation',
    seed=42
)

test_generator = valid_datagen.flow_from_directory(
    path_valid,
    target_size=(img_width, img_height),
    color_mode="rgb",
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    classes=image_list,
    subset='validation',
    seed=42
)

# Model definition
model = Sequential([
    Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, channels)),
    Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(filters // 2, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # ... (more convolutional blocks)

    Flatten(),
    Dense(filters, activation='relu'),
    Dropout(0.5),
    Dense(output_n, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Nadam(learning_rate=0.002), metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model with Early Stopping and Learning Rate Reduction
start_time = time.time()
for i in range(10):  # Train for 10 epochs (can be adjusted)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // batch_size,
        callbacks=[
            EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=9, verbose=1, mode='auto', min_delta=1e-6, cooldown=0, min_lr=1e-10)
        ]
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nEpoch {i+1} - Test accuracy: {test_acc:.4f}")

# Calculate the total training time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nElapsed Training Time: {elapsed_time:.2f} seconds")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Get predictions on the test set
filenames = test_generator.filenames
nb_samples = len(filenames)
predict = model.predict_generator(test_generator, steps=nb_samples)

# Save the model
model.save("model3.h5")
loaded_model = load_model("model3.h5")
loaded_model.summary()  # Verify that the loaded model is the same as the original
