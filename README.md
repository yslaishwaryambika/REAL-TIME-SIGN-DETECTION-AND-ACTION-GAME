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

Model:
The landmarks are fed into an LSTM model trained on sequences of gestures.
The LSTM predicts the gesture by analyzing the temporal pattern of movements.

Output:
The recognized gesture is displayed in real-time on the video feed.
![image](https://github.com/user-attachments/assets/ecc88d90-6534-4b48-b0b9-b2b6e9c2aa98)

# Dataset
Data Source: Custom dataset created by recording hand gestures using MediaPipe.
Gestures: Predefined gestures include:
"Hello"
"Thank You"
"Yes"
"No"
"I Love You"
The dataset includes 15 sequences for each gesture, with each sequence containing 15 frames.

# Usage
Sign Language Learning: Helps users learn and practice basic sign language gestures.
Accessibility: Can be extended to assist hearing or speech-impaired individuals.
Real-Time Interaction: Interactive system suitable for live applications.

# License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

