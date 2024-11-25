# REAL-TIME-SIGN-DETECTION-AND-ACTION-GAME
A real-time sign language detection system using LSTM and an interactive action game powered by CNN and LSTM-based gesture recognition.


# Project Description
This project implements a Sign Language Recognition System using Long Short-Term Memory (LSTM) networks. The system processes real-time hand gestures to classify them into predefined sign language gestures. Leveraging MediaPipe for hand landmark detection and LSTM for sequence modeling, the system provides a robust, lightweight, and real-time solution for recognizing gestures.

# Features
Real-Time Gesture Recognition: Detects and classifies hand gestures on live video feed.
Efficient Preprocessing: Utilizes MediaPipe to extract 3D hand landmarks (x, y, z coordinates).
Temporal Analysis: Uses LSTM to analyze sequences of hand movements.
Customizable Gestures: Trained on a dataset of predefined gestures,additional gestures can be added.
Lightweight and Fast: Designed for real-time applications with low latency.

# Technologies Used

•	Python: Programming language.

•	MediaPipe: For hand landmark detection.

•	TensorFlow/Keras: For building and training the LSTM model.

•	OpenCV: For handling video input and visualization.

•	Matplotlib: For visualizing results (if needed).

# Setup and Installation
1.Clone the repository

2.Install Required Libraries

3.Run the Code

# How it Works
Preprocessing: 
The system captures real-time video input using OpenCV.
MediaPipe extracts 21 hand landmarks (x, y, z coordinates) per frame.

Preprocessing for the Action Game :
Preprocessing for the action game involves extracting and normalizing hand landmarks from gesture images or videos using MediaPipe for consistent input to the model, Using both LSTM and CNN.

Model:
The landmarks are fed into an LSTM model trained on sequences of gestures.
The LSTM predicts the gesture by analyzing the temporal pattern of movements.

Model for The Action Game:
The action game leverages a hybrid model combining CNN for spatial feature extraction and LSTM for temporal sequence analysis of hand gestures.

Output:
The recognized gesture is displayed in real-time on the video feed.
![image](https://github.com/user-attachments/assets/8eda02b6-da2c-4208-a4fb-40b1fbf93f55)

Output For the Action Game:

![image](https://github.com/user-attachments/assets/55f84d27-11df-422d-84ed-473a0d8948b7)


# Dataset
Data Source: Custom dataset created by recording hand gestures using MediaPipe.
Gestures: Predefined gestures include:
"Hello"
"Thank You"
"Yes"
"No"
"I Love You"
The dataset includes 15 sequences for each gesture, with each sequence containing 15 frames.

# Dataset for Action Gane 
The dataset for the Sign Action Game includes sourced online data comprising gestures for letters, alphabets, and numbers. This ensures diversity and accuracy for training and real-time recognition.

# Usage
Sign Language Learning: Helps users learn and practice basic sign language gestures.
Accessibility: Can be extended to assist hearing or speech-impaired individuals.
Real-Time Interaction: Interactive system suitable for live applications.

The action game is an interactive application that uses real-time gesture recognition, allowing users to play by performing hand signs detected through CNN and LSTM models, promoting learning and engagement with sign language.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

