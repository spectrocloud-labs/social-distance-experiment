import cv2, threading, queue

class ThreadingClass:
  # initiate threading class
  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read the frames as soon as they are available, discard any unprocessed frames;
  # this approach removes OpenCV's internal buffer and reduces the frame lag
  def _reader(self):
    while True:
      ret = self.cap.grab() # read the frames
      if not ret:
        break

  def read(self):
    ret, frame = self.cap.retrieve()
    return frame

  def release(self):
    self.cap.release()
