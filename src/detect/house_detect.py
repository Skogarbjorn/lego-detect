import cv2
from lib.frame_grabber import FrameGrabber
import numpy as np
import os
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "..", "models", "houses-model", "best.pt")
CLASS_NAMES = ['SingularHouse', 'ApartmentComplex']
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

model = YOLO(MODEL_PATH)
#model.to("cpu")

def detect_houses(frame):
    results = model(frame, verbose=False)

    boxes = []
    confidences = []
    class_ids = []

    for r in results:
        for poly, conf, cls in zip(r.obb.xyxyxyxy, r.obb.conf, r.obb.cls):
            poly_np = poly.cpu().numpy().reshape(4, 2).astype(int)
            boxes.append(poly_np)
            confidences.append(float(conf))
            class_ids.append(int(cls))
    return boxes, confidences, class_ids

def draw_houses(frame, boxes, confidences, class_ids):
    for poly, conf, cls_id in zip(boxes, confidences, class_ids):
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"
        cv2.putText(frame, label, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

if __name__ == "__main__":
    grabber = FrameGrabber()
    while True:
        frame = grabber.read()
        if frame is None:
            continue

        boxes, confidences, class_ids = detect_houses(frame)

        draw_houses(frame, boxes, confidences, class_ids)

        cv2.imshow("path detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    grabber.release()
    cv2.destroyAllWindows()
