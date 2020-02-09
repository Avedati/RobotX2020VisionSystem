import chessboard as cb
import cv2
import math
import numpy as np
import os
import time
from tests import *

HEX = 'hexagon'
TAPE = 'vision_tape'

def GetEdgeImage(frame, sigma=0.33):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  v = np.median(gray)
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  lower = 100 #10
  upper = 160
  kernel = np.ones((5,5), np.uint8) 

  print(lower, upper)

  gaussian = cv2.GaussianBlur(frame, (7, 7), 0)
  ##gaussian = cv2.bilateralFilter(gray, 7, 10, 0)
  #canny = cv2.Canny(gaussian, lower, upper)
  #ret, canny = cv2.threshold(gaussian, 50, 255, cv2.THRESH_BINARY)
  #canny = cv2.adaptiveThreshold(
  #    gaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  canny0 = cv2.Canny(gaussian[:,:,0], lower, upper)
  canny1 = cv2.Canny(gaussian[:,:,1], lower, upper)
  canny2 = cv2.Canny(gaussian[:,:,2], lower, upper)
  canny = np.maximum(canny0, np.maximum(canny1, canny2))
  canny = cv2.dilate(canny, kernel, iterations=1)
  canny = cv2.erode(canny, kernel, iterations=1)
  return canny
  

def RunPoseEstimationTape(visionTape, calib):
  objectPoints = np.array((8, 2), np.float32)
  theta = np.pi
  for i in range(4):
   objectPoints[7-i,:] = (17.32 * np.cos(theta), 17.32 * np.sin(theta))
   theta += np.pi / 3 



  rvec = None
  tvec = None
  useExtrinsicGuess = False

  while True:
    frame = camera.GetFrame()
    if frame is None:
      break
    objectPoints, imagePoints = chess.GetObjectAndImagePoints(frame, draw=False)
    if objectPoints is not None:
      _, rvec, tvec = cv2.solvePnP(
          objectPoints, imagePoints, calib.cameraMatrix, calib.distCoeffs,
          rvec=rvec, tvec=tvec, useExtrinsicGuess=useExtrinsicGuess)
      useExtrinsicGuess = True

      # Draw checkerboard points.
      checkerboard, _ = cv2.projectPoints(
          objectPoints, rvec, tvec, calib.cameraMatrix, calib.distCoeffs)
      for point in checkerboard:
        point = np.squeeze(point)
        cv2.circle(frame, tuple(point), 4, (255, 0, 0), 2)

      # Draw coordinate axes.
      coordFrame.Draw(frame, rvec, tvec, calib)

      if not camera.OutputFrameAndTestContinue('SolvePnP', frame):
        break

class ObjectTracker(object):
  def __init__(self, objects):
    self.objectsToTrack = objects

  def Track(self, frame):
    edges = GetEdgeImage(frame)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [self.ExtractPolygon(c) for c in contours]
    vision_tapes = [v for v in [self.TestVisionTape(p) for p in contours] if v is not None]
    #hexagons = [TestHexagons(p) for p in polygons]

		RunPose		

    edges = cv2.drawContours(frame, vision_tapes, -1, (0, 255, 0), 1) #, hierarchy=hierarchy, maxLevel=1)
    for contour in vision_tapes:
      for i, point in enumerate(list(contour)):
        edges = cv2.putText(frame, str(i+1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    """
    edges = frame
    for idx, contour in enumerate(contours):
      #color = tuple((np.random.random((3)) * 255).astype(int).tolist())
      color = ( (idx * 37) % 255, (idx * 31) % 255, (idx * 41) % 255)
      edges = cv2.drawContours(frame, [contour], -1, color, 1)#, hierarchy=hierarchy, maxLevel=2)
    
    vision_tape
    for contour in enumerate(contours):
    """

    return edges


  def ExtractPolygon(self, contour):
    eps = 0.005 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, eps, True)
    return polygon

  def TestVisionTape(self, polygon):
    if True: #polygonTests(polygon):
      result1 = processPolygon(polygon, 8)
      if result1 is not None:
        result = visionTapeTest(result1)
        if result is not None:
          if not cv2.isContourConvex(polygon):
            if cv2.contourArea(polygon) >= 2000:
              return result
    return None

def main():
  dataDir = '/Users/spiderfencer/auton_2020_frc/test_powerports/test/res'
  videoSource = os.path.join(dataDir, 'vision-tape-target-1.mp4')
  #video_source = 0

  calibVideo = os.path.join(dataDir, 'chessboard-tv.mp4')
  imageHeight = 360
  maxSamples = 25

  calib = cb.Calibration(calibVideo, imageHeight, maxSamples)
  if not calib.LoadOrCompute():
    print('Could not load or compute calibration.')
    return

  outputFile = videoSource + '-' + calib.Id() + '-tracked.mp4'
  camera = cb.CameraSource(videoSource, calib.imageHeight, outputFile)

  objects = {}#HEX: Hexagon(), TAPE: VisionTape()}
  tracker = ObjectTracker(objects)

  coordFrame = cb.CoordinateFrame(10)

  while True:
    frame = camera.GetFrame()
    if frame is None:
      break
    
    edges = tracker.Track(frame)
    
    ret = True
    ret = ret and cb.ShowFrameAndTestContinue('Frame', frame)
    ret = ret and cb.ShowFrameAndTestContinue('Edges', edges)
    if not ret:
      break;

    continue
    objectPoints = np.zeros([0, 3], np.float32)
    imagePoints = np.zeros([0, 2], np.float32)
    imagePoints = None
    for key,polygon in polygons.items():
      polygon.ObjectPoitns()
    

"""
  print(ORIGINAL_WIDTH)
  nFrames = 0
  savedHexagons = []
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
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blurred = cv2.GaussianBlur(gray, (7, 7), 0)
      canny = cv2.Canny(blurred, 10, 160)
      #retval, canny = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
      #canny = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

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
"""

if __name__ == '__main__':
  main()
