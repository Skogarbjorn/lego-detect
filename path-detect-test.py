from ultralytics import YOLO
import cv2

from frame_grabber import FrameGrabber

model = YOLO("path-model/best.pt")  

url = "http://192.168.0.19:4747/video"
grabber = FrameGrabber(url)

while True:
    frame = grabber.read()
    if frame is None:
        continue

    results = model(frame)

    for r in results:
        for obb in r.obb:  
            xywh = obb.xywhr[0] 
            print("raw xywhr", obb.xywhr[0])
            conf = obb.conf[0]
            cls = int(obb.cls[0])

            if conf < 0.2:
                continue

            cx, cy, w, h, angle_rad = xywh.cpu().numpy()
            angle_deg = angle_rad * 180 / 3.1415926

            img_h, img_w = frame.shape[:2]
            print(f"Class {cls}, conf: {conf:.2f}, center: ({cx:.1f}, {cy:.1f}), size: ({w:.1f}, {h:.1f}), angle: {angle_deg:.1f}")

            rect = ((cx, cy), (w, h), angle_deg)
            box = cv2.boxPoints(rect).astype(int)
            cv2.polylines(frame, [box], isClosed=True, color=(0,255,0), thickness=2)

    cv2.imshow("path detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


grabber.release()
cv2.destroyAllWindows()
