Traffic Sign Prediction System
This project implements a traffic sign classification system that uses a deep learning model to predict traffic signs from images. The system includes a training phase, where a neural network model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, and a prediction phase, where users can upload traffic sign images to get predictions.

Project Overview
Traffic Sign Classification: The model is trained to recognize 10 categories of traffic signs, such as stop signs, speed limit signs, and pedestrian crossing signs.

Deep Learning Model: A Convolutional Neural Network (CNN) is used for image classification.

GUI Interface: A Tkinter-based GUI allows users to upload images and get predictions.

REQUIREMENTS 
Python 3.x
TensorFlow / Keras
OpenCV
Pillow (PIL)
NumPy
Tkinter

You can install the required packages using:
pip install tensorflow opencv-python pillow numpy

PROJECT STRUCTURE




![Screenshot 2025-03-31 110517](https://github.com/user-attachments/assets/cefbb1dd-20c6-4cf6-b6a1-f0c1815afe96)
















GETTING STARTED

Step 1: Download and Prepare the Dataset
-Download the GTSRB (German Traffic Sign Recognition Benchmark) dataset.
-Download GTSRB dataset
-Organize the dataset into separate folders for each traffic sign category (e.g., stop, speed_limit, etc.).

Step 2: Training the Model
-Train the Model:
-The traffic.py script is responsible for loading the dataset, preprocessing the images, and training the model.

You can run the training script by executing:
python MBINA/traffic.py

Model Output:
After training, the model will be saved as best_model.h5 in the project directory. This file is used for making predictions.

Step 3: Prepare Test Images
Collect 10 unique test images of traffic signs (not from the dataset), and store them in the images/ folder.

Example:
0_stop_1.png
1_speed_limit_50.png
2_yield.png

These images should represent different sizes and traffic signs.

Step 4: Running the GUI
Predict Traffic Sign:
The predict_sign.py script provides a GUI where users can upload an image, and the system will predict the traffic sign.

Run the Prediction GUI:
Launch the GUI by running:
-python MBINA/predict_sign.py
-Upload an Image:
-Click the "Upload Image" button to select an image file. The predicted class and confidence will be displayed along with the image.

Step 5: Understanding the Code
traffic.py: This file handles the training process. It reads and preprocesses the dataset, builds and trains the neural network model, and saves the trained model as best_model.h5.
predict_sign.py: This file provides the Tkinter-based graphical user interface (GUI) to upload images and get predictions from the model.

Step 6: Adjusting Categories
In the predict_sign.py script, there is a list of categories that correspond to the model's output. Make sure the list matches the number of classes in your dataset:

![Screenshot 2025-03-31 111005](https://github.com/user-attachments/assets/6e08e078-d1e5-4a0c-9725-f39b6890caaf)
If the model was trained on more or fewer categories, adjust this list accordingly.

Step 7: Model Evaluation
Once the model is trained, evaluate its performance on test images. If the accuracy is not satisfactory, consider:

Using a larger or better dataset
Fine-tuning the model

Applying data augmentation techniques
Troubleshooting

IndexError: list index out of range:
Ensure the categories list in the GUI script matches the model's output.

OpenCV error when resizing images:
Ensure that the image files are correctly read and are valid images.

Model not loading properly:
Make sure the best_model.h5 file is in the correct directory and matches the model used during training.

PROBLEMS ENCOUNTERED:
"I ENCOUNTERED A LONG LIST OF PROBLEMS WHICH MADE ME WASTE A LOT OF TIME FINISHING UP THE PROJECT ON TIME AND I REALLY HOPE MY COURSE INSRUCTOR MR. NGUH PRINCE UNDERSTANDS."
-

CONCLUSION
This project demonstrates the use of deep learning (CNNs) for traffic sign classification. The system is built using TensorFlow and Keras for model training and Tkinter for the GUI. By following the steps in this README, you can train your own model, make predictions, and even experiment with new datasets to improve the model's performance.

I NEVER HAD TIME TO DOWNLOAD ALL THE DIFFERENT IMAGE CATEGORIES SO I HAVE LES THAN TEN IMAGE CATEGORIES 
