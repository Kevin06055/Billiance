import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Directories
data_dir = 'data'
classes = ['Normal', 'Shoplifting']

# Parameters
max_people = 10  # Maximum number of people to handle in a single frame
num_keypoints = 33  # Number of key points detected by MediaPipe
feature_dim = 3  # x, y, visibility
num_frames = 20  # Number of frames per sequence

sequences = []
labels = []

for label, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for video_name in os.listdir(class_dir):
        video_path = os.path.join(class_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        pose_sequence = []
        frame_counter = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_counter += 1
            print(f'Processing frame {frame_counter}')
            
            # Convert the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            if results.pose_landmarks:
                pose_frame = []
                for lm in results.pose_landmarks.landmark:
                    pose_frame.extend([lm.x, lm.y, lm.visibility])
                
                pose_sequence.append(pose_frame)
                
                if len(pose_sequence) == num_frames:
                    sequences.append(np.array(pose_sequence))
                    labels.append(label)
                    pose_sequence = []

        cap.release()

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Save the data
np.save('sequences1.npy', sequences)
np.save('labels1.npy', labels)
