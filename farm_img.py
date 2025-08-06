import cv2
import os
from datetime import datetime
from frame_grabber import FrameGrabber

grabber = FrameGrabber()
folder_path = "images/"
os.makedirs(folder_path, exist_ok=True) 

while True:
    frame = grabber.read()
    if frame is None:
        continue

    now = datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".png"

    cv2.imshow('frame grabber cam', frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 32:
        cv2.imwrite(folder_path + filename, frame)
    if key == 27:
        break

cv2.destroyAllWindows()
