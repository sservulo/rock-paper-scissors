from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics import solutions

MODEL_PATH = (
    "/home/samuel/workspace/rock-paper-scissors/runs/detect/train4/weights/best.pt"
)


def get_winner(detections: list[int]):
    res = ""
    if len(detections) > 2:
        print("Too many!")
        return res

    readable = list(map(number_to_class, detections))
    readable.sort()

    if readable == ["Paper", "Rock"]:
        res = "Paper"
    elif readable == ["Rock", "Scissor"]:
        res = "Rock"
    elif readable == ["Paper", "Scissor"]:
        res = "Scissor"

    return res


def number_to_class(num: int):
    if num == 0:
        return "Paper"
    elif num == 1:
        return "Rock"
    return "Scissor"


# Load a model
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

w, h = [int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]

region_points = [(0, 0), (0, h / 3), (w, h / 3), (w, 0)]

tracker = solutions.TrackZone(
    show=True,  # display the output
    region=region_points,  # pass region points
    model=MODEL_PATH,  # use any model that Ultralytics support, i.e. YOLOv9, YOLOv10
    # line_width=2,  # adjust the line width for bounding boxes and text display
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = tracker.process(frame)
    # print("Results: {}".format(results))
    print("Tracker: {}".format(tracker.clss))
    winner = get_winner(tracker.clss)
    if winner != "":
        print("{} is the winner!".format(winner))

    # # Run YOLO11 tracking on the frame, persisting tracks between frames
    # results = model.track(frame, persist=True)

    # # Visualize the results on the frame
    # annotated_frame = results[0].plot()
    # # print(results[0])

    # # Display the captured frame
    # cv2.imshow("Camera", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
