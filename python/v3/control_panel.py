import numpy as np
import cv2
import math
import time

WIDTH = -1
HEIGHT = -1

class Vector2:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def dot(self, other):
    return self.x * other.x + self.y * other.y

  def length(self):
    return math.sqrt(self.x * self.x + self.y * self.y)

def dist(x1, y1, x2, y2):
  return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


def DetectYellow(hsv):
    # Threshold the HSV image, keep only the color pixels
    mask = cv2.inRange(hsv, (25, 200, 100), (30, 255, 255))
    return np.atleast_3d(mask)/255.0


def DetectGreen(hsv):
    # Threshold the HSV image, keep only the color pixels
    mask = cv2.inRange(hsv, (36, 100, 50), (86, 255, 255))
    return np.atleast_3d(mask)/255.0

def DetectBlue(hsv):
    # Threshold the HSV image, keep only the color pixels
    mask = cv2.inRange(hsv, (100, 100, 100), (140, 255, 255))
    return np.atleast_3d(mask)/255.0

def DetectRed(hsv):
    # Threshold the HSV image, keep only the red pixels
    lower_red = cv2.inRange(hsv, (  0, 100, 100), ( 10, 255, 255))
    upper_red = cv2.inRange(hsv, (170, 100, 100), (179, 255, 255)) 
    red = lower_red + upper_red
    return (np.atleast_3d(red)/255.0)

def main():
  camera = cv2.VideoCapture("/Users/kwatra/Home/pvt/robotx/RobotX2020VisionSystem/data/controlpanel.mp4")
  #camera = cv2.VideoCapture(0)
  ORIGINAL_WIDTH = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
  ORIGINAL_HEIGHT = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
  HEIGHT = min(ORIGINAL_HEIGHT, 360)
  WIDTH = int(ORIGINAL_WIDTH * HEIGHT / ORIGINAL_HEIGHT)
  FPS = camera.get(cv2.CAP_PROP_FPS)
  print(FPS)
  print(ORIGINAL_WIDTH)
  nFrames = 0
  frames = []
  if camera.isOpened():
    start_time = time.time()
    while True:
      frame_time = time.time() 
      if nFrames > 0:
       print('FPS: ', nFrames / (frame_time - start_time) )
      nFrames += 1
      ret, frame = camera.read()
      if frame is None:
        break

      frame = cv2.resize(frame, (WIDTH, HEIGHT))
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      red = DetectRed(hsv) *       [0, 0, 1]
      blue = DetectBlue(hsv) *     [1, 0, 0]
      green = DetectGreen(hsv) *   [0, 1, 0]
      yellow = DetectYellow(hsv) * [0, 1, 1]
      panel = red + blue + green + yellow
      cv2.imshow('panel', panel)

      gray_red = DetectRed(hsv) * 0.25
      gray_blue = DetectBlue(hsv) * 0.5
      gray_green = DetectGreen(hsv) * 0.75
      gray_yellow = DetectYellow(hsv) * 1 
      gray_panel = gray_red + gray_blue + gray_green + gray_yellow
      gray_panel = (gray_panel * 255).astype(np.uint8)

      #gray = cv2.cvtColor((gray_panel * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
      cv2.imshow('gray', gray_panel)

      blurred = cv2.GaussianBlur(gray_panel, (3, 3), 0)
      canny = cv2.Canny(blurred, 10, 160)
      cv2.imshow('canny', canny)


      k = cv2.waitKey(1) & 0xFF
      if k == 27:
        break
      continue

        

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blurred = cv2.GaussianBlur(gray, (7, 7), 0)
      canny = cv2.Canny(blurred, 10, 160)

      kernel = np.ones((3, 3), np.uint8)
      dilated = cv2.dilate(canny, kernel)
      eroded = cv2.erode(dilated, kernel)
      contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      hexagons = []
      octagons = []
        
      for i in range(len(contours)):
        cnt = contours[i]
        epsilon = 0.005*cv2.arcLength(cnt, True)
        polygon = cv2.approxPolyDP(cnt, epsilon, True)
        result = polygonTests(polygon)
        if result == -1:
          continue
        elif result >= 6:
          hexagonPair = processHexagon(polygon)
          if not cv2.isContourConvex(hexagonPair[1]):
            continue
          hexagons.append(hexagonPair)
        elif result >= 8:
          octagonPair = processOctagon(polygon)
          if octagonPair is not None:
            if not cv2.isContourConvex(octagonPair[1]):
              continue
            octagons.append(octagonPair)

      hexagons.sort(key=lambda h: h[0])
      octagons.sort(key=lambda o: o[0])
      
      hexagonsToDraw = [pair[1] for pair in hexagons]
      octagonsToDraw = [pair[1] for pair in octagons]

      #if nFrames % 100 == 0:
      #  savedHexagons.append(hexagonsToDraw[0])
      #  frames.append(frame.copy())
      
      print(len(hexagonsToDraw))

      for i in range(min(1, len(hexagonsToDraw))):
        cv2.drawContours(frame, hexagonsToDraw, i, (0, 255, 0), 1)
        for j in range(len(hexagonsToDraw[i])):
          cv2.putText(frame, str(j+1), tuple(hexagonsToDraw[i][j]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
      for i in range(min(1, len(octagonsToDraw))):
        cv2.drawContours(frame, octagonsToDraw, i, (0, 0, 255), 1)

      viz = cv2.resize(eroded, None, fx=0.5, fy=0.5)
      cv2.imshow('frame', viz)
      k = cv2.waitKey(1) & 0xFF
      if k == 27:
        break
      continue


      #if nFrames >= 1000:
      #  break

    #objectsPoints = []
    pts = []

    theta = math.pi * 4 / 3;
    for i in range(6):
      pts.append((15 * math.cos(theta), 15 * math.sin(theta), 0))
      theta += math.pi / 3

    for i in range(10):
      objectPoints.add(np.array(pts))

    _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, savedHexagons, (WIDTH, HEIGHT))
    nFrames = 0
    imageIndex = 0
    while True:
      nFrames += 1
      if nFrames % 5000 == 0:
        imageIndex += 1
      if nFrames >= 50000:
        break
      img = frames[imageIndex]
      dMat = cv2.projectPoints(objectPoints[imageIndex], rvecs[imageIndex], tvecs[imageIndex], cameraMatrix, distCoeffs)
      cv2.drawContours(img, [dMat], 0, (0, 0, 255), 4)
      cv2.imshow('test', img)

if __name__ == '__main__':
  main()
