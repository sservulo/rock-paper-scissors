from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load a model
model = YOLO(
    "/home/samuel/workspace/rock-paper-scissors/runs/detect/train3/weights/best.pt"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    # print(results[0])

    # Display the captured frame
    cv2.imshow("Camera", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
