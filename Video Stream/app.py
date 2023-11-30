from flask import Flask, Response, render_template
import cv2
import torch
import mediapipe as mp
import torch.nn as nn
from collections import deque
import os

class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FallDetectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        out  = torch.sigmoid(out)
        return out

input_size=132
hidden_size=132
num_layers=3
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    model_complexity=1, 
                    enable_segmentation=False, 
                    min_detection_confidence=0.5)

model = FallDetectionLSTM(input_size, hidden_size, num_layers)
model.load_state_dict(torch.load('/Users/varunshankarhoskere/Desktop/Academics/Extra/lstm_model.pth'))
model.eval()



app = Flask(__name__)

def gen_frames():
    camera=cv2.VideoCapture("output.mp4")
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concatenate frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import cv2
import torch
from collections import deque

# Initialize VideoCapture
cap = cv2.VideoCapture("/Users/varunshankarhoskere/Downloads/WhatsApp Video 2023-11-22 at 12.45.34.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
ten_seconds_frames = int(10 * fps)  # Calculate the number of frames for 10 seconds

frame_window = deque(maxlen=ten_seconds_frames)
fall_detected = False  # Flag to check if fall is already detected

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is captured

    frame_window.append(frame)
    
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Extract pose landmarks and convert to tensor
        pose_landmarks = torch.tensor([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        pose_landmarks = pose_landmarks.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions

        # Make prediction using the LSTM model
        with torch.no_grad():
            output = model(pose_landmarks)
            predicted_label = (output > 0.5).float().item()

    if predicted_label == 1 and not fall_detected:
        # Save the last 10 seconds of frames
        fall_detected = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))
        for f in frame_window:
            out.write(f)
        out.release()

    # Rest of your code for displaying the frame and breaking the loop...

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
