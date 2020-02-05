import cv2
import math
import numpy as np
import os
import time


class CameraSource(object):
  def __init__(self, camera_source, output_file=None):
    self.camera = cv2.VideoCapture(camera_source)
    self.ORIGINAL_WIDTH = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.ORIGINAL_HEIGHT = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.HEIGHT = min(self.ORIGINAL_HEIGHT, 360)
    self.WIDTH = int(self.ORIGINAL_WIDTH * self.HEIGHT / self.ORIGINAL_HEIGHT)
    self.startTime = time.time()
    self.nFrames = 0
    if output_file:
      self.writer = cv2.VideoWriter(
              output_file,
              cv2.VideoWriter_fourcc(*'H264'),
              25,
              (self.WIDTH, self.HEIGHT))
    else:
        self.writer = None
    print(self.ORIGINAL_WIDTH)

  def GetFrame(self):  
    frame = None
    if self.camera.isOpened():
      frameTime = time.time() 
      if self.nFrames > 0 and self.nFrames % 50 == 0:
        print('FPS: ', self.nFrames / (frameTime - self.startTime))
      self.nFrames += 1
      ret, frame = self.camera.read()
      if frame is not None:
        frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
    return frame

  def ImageSize(self):
    return (self.WIDTH, self.HEIGHT)

  def OutputFrameAndTestContinue(self, message, frame):
    if self.writer:
      self.writer.write(frame)
    cv2.imshow(message, frame)
    k = cv2.waitKey(1) & 0xFF
    return k != 27

  def __del__(self):
    if self.writer:
      self.writer.release()
    self.camera.release()
    cv2.destroyAllWindows()
        

class Chessboard(object):
  def __init__(self, rows=6, cols=8, squareWidth=8.0):
    self.chessPoints = np.zeros((rows * cols, 3), np.float32)
    self.chessPoints[:,:2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1,2) * squareWidth)
    self.patternSize = (cols, rows)
    self.squareWidth = squareWidth

  def GetObjectAndImagePoints(self, frame, draw=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, self.patternSize)
    if found:
      cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
      if draw:
        cv2.drawChessboardCorners(frame, self.patternSize, corners, found)
      return self.chessPoints, corners
    else:
      return None, None

  def SquareWidth(self):
    return self.squareWidth


def RunCalibration(calibVideo, maxSamples=0, forceRecompute=False):
  nChessFrames = 0
  objectPoints = []
  imagePoints = []

  camera = CameraSource(calibVideo)
  chess = Chessboard()

  height = camera.ImageSize()[1]
  id = str(height) + '-' + str(maxSamples)
  calibFileCam = calibVideo + '-' + str(id) + '-calib_cam.txt'
  calibFileDist = calibVideo + '-' + str(id) + '-calib_dist.txt'
    
  print('Calibration files:')
  print(calibFileCam)
  print(calibFileDist)

  if forceRecompute:
    print('Forcing recomputation of calibration data.')
 
  loadedCalib = False
  if (not forceRecompute and os.path.exists(calibFileCam) and
        os.path.exists(calibFileDist)):
    print('Loading calibratin info from files.')
    cameraMatrix = np.loadtxt(calibFileCam)
    success = cameraMatrix.shape == (3, 3)
    if success:
      distCoeffs = np.loadtxt(calibFileDist)
      success = distCoeffs.shape == (5,)
    if success:
      loadedCalib = True
      print('Loaded successfully.')
      print('CameraMatrix:\n', cameraMatrix)
      print('DistCoeffs:\n', distCoeffs)
      return cameraMatrix, distCoeffs
    print('Failed to load calibration from files.')

  print('Extracting frames for calibration.')
  while True:
    frame = camera.GetFrame()
    if frame is None:
      break
    if maxSamples > 0 and len(imagePoints) >= maxSamples:
      break

    calibSample = False
    objectPointsFrame, imagePointsFrame = chess.GetObjectAndImagePoints(
        frame, draw=True)
    if objectPointsFrame is not None:
      nChessFrames += 1
      if nChessFrames % 100 == 0:
        calibSample = True
        objectPoints.append(objectPointsFrame)
        imagePoints.append(imagePointsFrame)
        cv2.putText(frame, 'CALIBRATION SAMPLE', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        print('Extracted calibration sample ', len(imagePoints))
           
    # Display frame.
    if not camera.OutputFrameAndTestContinue('chess', frame):
      break
    if calibSample:
      time.sleep(0.5)
  
  print('Calibrating...')
  ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
      objectPoints, imagePoints, camera.ImageSize(), None, None) 
  
  print('Saving calibratin file ', calibFileCam)
  print(cameraMatrix)
  np.savetxt(calibFileCam, cameraMatrix)
  print('Saving calibratin file ', calibFileDist)
  print(distCoeffs)
  np.savetxt(calibFileDist, distCoeffs)
  print('Done.')

  return cameraMatrix, distCoeffs


def RunPoseEstimation(video, cameraMatrix, distCoeffs):
  output_file = video + '-detected_pose.mp4'
  camera = CameraSource(video, output_file)
  chess = Chessboard()
  
  coordFrame = np.zeros((4, 3), np.float32)
  coordFrame[0, ...] = [0, 0, 0]
  coordFrame[1, ...] = [1, 0, 0]
  coordFrame[2, ...] = [0, 1, 0]
  coordFrame[3, ...] = [0, 0, -1]
  coordFrame *= chess.SquareWidth() * 2

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
          objectPoints, imagePoints, cameraMatrix, distCoeffs,
          rvec=rvec, tvec=tvec, useExtrinsicGuess=useExtrinsicGuess)
      # Combine checkerboard points and coordinate axes points.
      objectPoints = np.concatenate((objectPoints, coordFrame), 0)
      useExtrinsicGuess = True
      projectedPoints, _ = cv2.projectPoints(
          objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
      # Draw checkerboard points.
      checkerboard = projectedPoints[0:-4]
      for point in checkerboard:
        point = np.squeeze(point)
        cv2.circle(frame, tuple(point), 4, (255, 0, 0), 2)
      # Draw coordinate axes.
      coords = projectedPoints[-4:]
      coords = [tuple(np.squeeze(x).astype(int)) for x in coords.tolist()]
      cv2.arrowedLine(frame, coords[0], coords[1], (  0, 255, 0), 2)
      cv2.arrowedLine(frame, coords[0], coords[2], (255, 0, 255), 2)
      cv2.arrowedLine(frame, coords[0], coords[3], (0, 255, 255), 2)
      if not camera.OutputFrameAndTestContinue('SolvePnP', frame):
        break


def main():
  calibVideo = '/Users/kwatra/Home/pvt/robotx/RobotX2020VisionSystem/data/chessboard-tv.mp4'
  cameraMatrix, distCoeffs = RunCalibration(calibVideo, maxSamples=25)
  RunPoseEstimation(calibVideo, cameraMatrix, distCoeffs)


if __name__ == '__main__':
  main()
