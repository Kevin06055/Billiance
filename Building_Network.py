import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking

# Parameters
num_keypoints = 33  # Number of key points detected by MediaPipe
feature_dim = 3  # x, y, visibility
num_frames = 20  # Number of frames per sequence

# Load sequences and labels
sequences = np.load('sequences1.npy')
labels = np.load('labels1.npy')

# Model Definition
input_pose = Input(shape=(num_frames, num_keypoints * feature_dim))
masked_input = Masking(mask_value=0.0)(input_pose)
x = LSTM(128, return_sequences=False)(masked_input)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=input_pose, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(sequences, labels, epochs=80, validation_split=0.2)

# Save the model
model.save('act.keras')
