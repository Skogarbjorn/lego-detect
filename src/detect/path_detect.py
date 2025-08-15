import os
from ultralytics import YOLO
import cv2
from lib.frame_grabber import FrameGrabber
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "..", "models", "path-model", "best.pt")

model = YOLO(MODEL_PATH) 
model.to("cpu")

def detect_paths(frame):
    results = model(frame, verbose=False)

    boxes = []
    for r in results:
        for obb in r.obb:  
            xy = obb.xyxyxyxy[0].cpu().numpy().astype(int)

            box = xy.reshape(4,2)
            boxes.append(box)
    return boxes

if __name__ == "__main__":
    grabber = FrameGrabber()

    while True:
        frame = grabber.read()
        if frame is None:
            continue

        boxes = detect_paths(frame)
        for box in boxes:
            cv2.polylines(frame, [box], isClosed=True, color=(0,255,0), thickness=2)

        cv2.imshow("path detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    grabber.release()
    cv2.destroyAllWindows()
