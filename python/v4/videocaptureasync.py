# file: videocaptureasync.py
import threading
import cv2

class VideoCaptureAsync:
    def __init__(self, src=0): # width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            grabbed = self.grabbed
            frame = self.frame.copy() if grabbed else None
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def release(self):
        self.stop()
        self.cap.release()

    def __exit__(self, exec_type, exc_value, traceback):
        self.release()
