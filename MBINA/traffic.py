import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Define target size for images
IMG_SIZE = 32

# Path to your downloaded dataset folder
DATASET_PATH = r"C:\Users\user\Desktop\Traffic\Traffic_Dataset"  # Use raw string for paths

# Categories mapping (modify based on your dataset)
categories = ['stop', 'speed_limit', 'yield', 'pedestrian_crossing', 'no_entry', 'turn_left','turn_right']
category_map = {category: idx for idx, category in enumerate(categories)}

# Function to preprocess images
def preprocess_images():
    images = []
    labels = []

    # Loop through the dataset
    for category in categories:
        category_path = os.path.join(DATASET_PATH, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to match model input size
            img = img / 255.0  # Normalize pixel values to [0, 1]
            
            images.append(img)
            labels.append(category_map[category])  # Label according to category
    
    return np.array(images), np.array(labels)

# Load and preprocess the dataset
images, labels = preprocess_images()

# One-hot encode labels
labels = to_categorical(labels, num_classes=len(categories))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build and compile the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(categories), activation='softmax')  # Output layer for each category
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model to a file
model.save("best_model.h5")

