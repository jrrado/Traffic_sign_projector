import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# Constants
IMG_SIZE = 32
MODEL_PATH = r"best_model.h5"  # Path to the saved model

# Load the trained model
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# Categories for prediction
categories = ['stop', 'speed_limit', 'yield', 'pedestrian_crossing', 'no_entry', 'turn_left']

# Function to preprocess and predict the image
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to model input size
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values to [0, 1]
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_id, confidence

# Function to open the image and make predictions
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        class_id, confidence = predict_image(file_path)
        result_label.config(text=f"Predicted Class: {categories[class_id]}\nConfidence: {confidence:.2f}")
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

# Build the Tkinter GUI
root = tk.Tk()
root.title("Traffic Sign Predictor")

btn = tk.Button(root, text="Upload Image", command=open_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Prediction will appear here")
result_label.pack()

root.mainloop()