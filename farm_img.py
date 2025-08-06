import cv2
from datetime import datetime
import threading

url = "http://192.168.0.19:4747/video"

class FrameGrabber:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame

    def read(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

grabber = FrameGrabber(url)

count = 0

while True:
    frame = grabber.read()
    if frame is None:
        continue

    now = datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".png"

    cv2.imshow('frame grabber cam', frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 32:
        cv2.imwrite("images/" + filename, frame)
    if key == 27:
        break

cv2.destroyAllWindows()
