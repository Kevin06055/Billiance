import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load the model
yolo = YOLO('yolov10s.pt')
model = load_model('act.keras')
# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Settings
num_keypoints = 33
feature_dim = 3
num_frames = 20
fps = 15
frame_interval = 1/fps

class ActivityRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Activity Recognition")

        # Set canvas size
        self.canvas_width = 640
        self.canvas_height = 480

        # Create GUI components
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.btn_real_time = tk.Button(root, text="Real-Time", command=self.start_real_time)
        self.btn_real_time.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_load_video = tk.Button(root, text="Load Video", command=self.load_video)
        self.btn_load_video.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_exit = tk.Button(root, text="Exit", command=root.quit)
        self.btn_exit.pack(side=tk.RIGHT, padx=10, pady=10)

        self.cap = None
        self.pose_sequence = []

    def start_real_time(self):
        self.cap = cv2.VideoCapture(0)
        self.process_stream()

    def load_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.process_stream()
        else:
            messagebox.showerror("Error", "Failed to load video.")

    def process_stream(self):
        prev_time = time.time()
        last_label = None

        while self.cap.isOpened():
            current_time = time.time()
            elapsed_time = current_time - prev_time

            if elapsed_time >= frame_interval:
                prev_time = current_time

                ret, frame = self.cap.read()
                if not ret:
                    break

                # Resize the frame to fit the canvas
                frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    pose_frame = []
                    min_x, min_y = 1, 1
                    max_x, max_y = 0, 0

                    for lm in results.pose_landmarks.landmark:
                        min_x = min(min_x, lm.x)
                        min_y = min(min_y, lm.y)
                        max_x = max(max_x, lm.x)
                        max_y = max(max_y, lm.y)
                        pose_frame.extend([lm.x, lm.y, lm.visibility])

                    self.pose_sequence.append(pose_frame)

                    if len(self.pose_sequence) == num_frames:
                        pose_sequence_np = np.expand_dims(np.array(self.pose_sequence), axis=0)
                        prediction = model.predict(pose_sequence_np)
                        predicted_class = np.argmax(prediction)

                        current_label = 'Shoplifting' if predicted_class == 1 else 'Normal'
                        print(current_label)

                        if current_label != last_label:
                            last_label = current_label

                        self.pose_sequence = []

                if last_label:
                    box_color = (0, 0, 255) if last_label == 'Shoplifting' else (0, 255, 0)
                    cv2.putText(frame_bgr, f'{last_label}', 
                                (int(min_x * frame.shape[1]), int(min_y * frame.shape[0]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame_bgr, 
                                  (int(min_x * frame.shape[1]), int(min_y * frame.shape[0])),
                                  (int(max_x * frame.shape[1]), int(max_y * frame.shape[0])),
                                  box_color, 2)

                # Convert frame to ImageTk format for display in Tkinter canvas
                img = Image.fromarray(frame_bgr)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.root.update_idletasks()

                # Display the frame using OpenCV
                cv2.imshow('Activity Recognition - OpenCV', frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ActivityRecognitionApp(root)
    root.mainloop()
