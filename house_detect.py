import cv2
import numpy as np

MODEL_PATH = 'houses-model/best.onnx'
CLASS_NAMES = ['SingularHouse', 'ApartmentComplex']
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

net = cv2.dnn.readNetFromONNX(MODEL_PATH)

def detect_houses(frame):
    original_height, original_width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward()[0]  

    boxes = []
    confidences = []
    class_ids = []

    for row in outputs:
        cx, cy, w, h = row[:4]
        obj_conf = row[4]
        class_scores = row[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]

        confidence = obj_conf * class_conf
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        
        x = int((cx - w / 2) * original_width / INPUT_WIDTH)
        y = int((cy - h / 2) * original_height / INPUT_HEIGHT)
        w = int(w * original_width / INPUT_WIDTH)
        h = int(h * original_height / INPUT_HEIGHT)

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    return (boxes, confidences, indices, class_ids)

def draw_houses(frame, boxes, confidences, indices, class_ids):
    for i in indices:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        label = f'{CLASS_NAMES[class_id]}: {confidences[i]:.2f}'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
