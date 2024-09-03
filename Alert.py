import os
import pickle
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import tkinter as tk
from tkinter import filedialog, messagebox  # Import filedialog and messagebox
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from PIL import Image, ImageTk
from ultralytics import YOLO

from random import randint

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
OUR_EMAIL = 'kkevinjc328@gmail.com'  # Replace with your email

def gmail_authenticate():
    """Authenticate and create a service for Gmail API."""
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('jason.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def create_message(to, subject, body):
    """Create an email message."""
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = OUR_EMAIL
    message['subject'] = subject

    msg_body = MIMEText(body)
    message.attach(msg_body)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw_message}

def send_message(service, to, subject, body):
    """Send an email message."""
    message = create_message(to, subject, body)
    try:
        message = service.users().messages().send(userId='me', body=message).execute()
        print(f'Sent message to {to} Message Id: {message["id"]}')
    except Exception as error:
        print(f'An error occurred: {error}')

# Activity Recognition Application
class ActivityRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Activity Recognition")

        # Create GUI components
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.btn_real_time = tk.Button(root, text="Real-Time", command=self.start_real_time)
        self.btn_real_time.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_load_video = tk.Button(root, text="Load Video", command=self.load_video)
        self.btn_load_video.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_exit = tk.Button(root, text="Exit", command=self.root.quit)
        self.btn_exit.pack(side=tk.RIGHT, padx=10, pady=10)

        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        # Load the model
        self.yolo = YOLO('yolov10s.pt')
        self.model = load_model('activity_recognition_model.keras')

        # Settings
        self.num_keypoints = 33
        self.feature_dim = 3
        self.num_frames = 20
        self.fps = 15
        self.frame_interval = 1 / self.fps

        self.cap = None
        self.pose_sequence = []
        self.shoplifting_start_time = None  # Track when shoplifting starts
        self.shoplifting_duration_threshold = 4  # Duration threshold in seconds

    def start_real_time(self):
        if self.cap is not None:
            self.cap.release()  # Release any existing video capture
        self.cap = cv2.VideoCapture(0)
        self.process_stream()

    def load_video(self):
        if self.cap is not None:
            self.cap.release()  # Release any existing video capture
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

            if elapsed_time >= self.frame_interval:
                prev_time = current_time

                ret, frame = self.cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image)
                frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    pose_frame = []
                    min_x, min_y = 1, 1
                    max_x, max_y = 0, 0

                    for lm in results.pose_landmarks.landmark:
                        min_x = min(min_x, lm.x)
                        min_y = min(min_y, lm.y)
                        max_x = max(max_x, lm.x)
                        max_y = max(max_y, lm.x)
                        pose_frame.extend([lm.x, lm.y, lm.visibility])

                    self.pose_sequence.append(pose_frame)

                    if len(self.pose_sequence) == self.num_frames:
                        pose_sequence_np = np.expand_dims(np.array(self.pose_sequence), axis=0)
                        prediction = self.model.predict(pose_sequence_np)
                        predicted_class = np.argmax(prediction)

                        current_label = 'Shoplifting' if predicted_class == 1 else 'Normal'
                        print(current_label)

                        if current_label != last_label:
                            last_label = current_label
                            if current_label == 'Shoplifting':
                                self.shoplifting_start_time = current_time
                            else:
                                self.shoplifting_start_time = None

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

                # Check if shoplifting has persisted for more than the threshold duration
                if self.shoplifting_start_time and (current_time - self.shoplifting_start_time) >= self.shoplifting_duration_threshold:
                    self.send_email()  # Automatically send an email
                    self.shoplifting_start_time = None  # Reset the start time

                # Convert frame to ImageTk format for display
                img = Image.fromarray(frame_bgr)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.root.update_idletasks()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()  # Ensure that OpenCV windows are destroyed

    def send_email(self):
        # Authenticate and send the email
        service = gmail_authenticate()
        send_message(service, "kkevinjc328@gmail.com", "Shoplifting detected!!", 
                     f"A shoplifting event has been detected by the system at aisle number {randint(1,15)}.")
        messagebox.showinfo("Success", "Email sent successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ActivityRecognitionApp(root)
    root.mainloop()

