import cv2
import threading

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

