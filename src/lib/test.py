import cv2
import os
from datetime import datetime
from lib.frame_grabber import FrameGrabber
import sys

grabbers = []
for i in range(1, len(sys.argv)):
    grabber = FrameGrabber(sys.argv[i])
    grabbers.append(grabber)
if len(sys.argv) == 1:
    grabbers.append(FrameGrabber())

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(CURRENT_DIR, "..", "..", "misc", "farmed_images")

os.makedirs(folder_path, exist_ok=True) 

while True:
    frame = grabbers[0].read()
    if frame is None:
        continue

    now = datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".png"

    cv2.imshow('frame grabber cam', frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 32:
        cv2.imwrite(os.path.join(folder_path, filename), frame)
    if key == 27:
        break

cv2.destroyAllWindows()
