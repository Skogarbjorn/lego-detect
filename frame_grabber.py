import cv2
import threading

url = "http://192.168.0.19:4747/video"

class FrameGrabber:
    def __init__(self, src=url):
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

