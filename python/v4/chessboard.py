import cv2
import math
import numpy as np
import os
import time


def ShowFrameAndTestContinue(message, frame, height=None):
  if height is not None:
    width = int(frame.shape[1] * height / frame.shape[0])
    frame = cv2.resize(frame, (width, height))
  cv2.imshow(message, frame)
  k = cv2.waitKey(1) & 0xFF
  return k != 27, k


class VideoWriter(object):
  def __init__(self, output_file):
    self.writer = None
    self.output_file = output_file

  
  def OutputFrame(self, frame):
    if not self.output_file:
      return
    if self.writer is None:
      self.writer = cv2.VideoWriter(
          self.output_file,
          cv2.VideoWriter_fourcc(*'H264'),
          25,
          (frame.shape[1], frame.shape[0]))
    self.writer.write(frame)


  def __del__(self):
    if self.writer:
      self.writer.release()
 


class CameraSource(object):
  def __init__(self, cameraSource, height, output_file=None, startFrame=0):
    self.camera = cv2.VideoCapture(cameraSource)
    self.ORIGINAL_WIDTH = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.ORIGINAL_HEIGHT = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.HEIGHT = min(self.ORIGINAL_HEIGHT, height)
    self.WIDTH = int(self.ORIGINAL_WIDTH * self.HEIGHT / self.ORIGINAL_HEIGHT)
    self.startFrame = startFrame
    self.nFrames = 0
    self.writer = VideoWriter(output_file)


  def GetFrame(self):  
    # Processing on first call. 
    if self.nFrames == 0:
      # Skip some frames if requested.
      if self.startFrame > 0:
        skippedFrames = 0
        while self.camera.isOpened():
          ret, frame = self.camera.read()
          if frame is None:
            print('No more frames')
            return None
          skippedFrames += 1
          if skippedFrames >= self.startFrame:
            break
      # Start timer for first frame.
      self.startTime = time.time()

    # Get frame.
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


  def OutputFrameAndTestContinue(self, message, frame, height=None):
    self.writer.OutputFrame(frame)
    return ShowFrameAndTestContinue(message, frame, height)


  def __del__(self):
    self.camera.release()
    cv2.destroyAllWindows()
        


class Chessboard(object):
  def __init__(self, squareWidth, rows, cols):
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



class Calibration(object):
  def __init__(self,
               calibVideo,
               imageHeight,
               maxSamples,
               startFrame=0):
    self.imageHeight = imageHeight
    self.calibVideo = calibVideo
    self.maxSamples = maxSamples
    self.startFrame = startFrame
    name = self.Id()
    self.calibFileCam = calibVideo + '-' + str(name) + '-calib_cam.txt'
    self.calibFileDist = calibVideo + '-' + str(name) + '-calib_dist.txt'
    self.hasCalib = False
    self.cameraMatrix = None
    self.distCoeffs = None

  def Id(self):
    return str(self.imageHeight) + '-' + str(self.maxSamples)

  def PrintInfo(self):
    print('Calibration files:')
    print(self.calibFileCam)
    print(self.calibFileDist)
    if self.hasCalib:
      print('CameraMatrix:\n', self.cameraMatrix)
      print('DistCoeffs:\n', self.distCoeffs)
    else:
      print('Calibration not loaded or computed.')


  def LoadFromFile(self):
    if self.hasCalib:
      return True
    
    # Read from file.
    self.hasCalib = False
    if (os.path.exists(self.calibFileCam) and
        os.path.exists(self.calibFileDist)):
      print('Loading calibratin info from files.')
      self.cameraMatrix = np.loadtxt(self.calibFileCam)
      success = self.cameraMatrix.shape == (3, 3)
      if success:
        self.distCoeffs = np.loadtxt(self.calibFileDist)
        success = self.distCoeffs.shape == (5,)
        if success:
          self.hasCalib = True
          print('Loaded successfully.')
          self.PrintInfo()
          return True

    # Failed to read.
    if not self.hasCalib:
      print('Failed to load calibration from files.')
      self.cameraMatrix = None
      self.distCoeffs = None
      return False


  def LoadOrCompute(self,
                    squareWidth=None,
                    rows=None,
                    cols=None,
                    forceRecompute=False):
    if forceRecompute:
      print('Forcing recomputation of calibration data.')
    elif self.LoadFromFile():
      return True

    if squareWidth == None or rows == None or cols == None:
      raise ValueError(
          'Need to pass chessboard params to compute calibration')

    nChessFrames = 0
    objectPoints = []
    imagePoints = []
 
    camera = CameraSource(
        self.calibVideo, self.imageHeight, startFrame=self.startFrame)
    chess = Chessboard(squareWidth, rows, cols)
 
    print('Extracting frames for calibration.')
    while True:
      frame = camera.GetFrame()
      if frame is None:
        break
      if self.maxSamples > 0 and len(imagePoints) >= self.maxSamples:
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
      if not camera.OutputFrameAndTestContinue('chess', frame, height=None)[0]:
        print('User stopped calibration process...')
        return False
      if calibSample:
        time.sleep(0.5)
    
    print('Calibrating...')
    _, self.cameraMatrix, self.distCoeffs, _, _ = cv2.calibrateCamera(
        objectPoints, imagePoints, camera.ImageSize(), None, None) 
    self.hasCalib = True
    
    self.PrintInfo()
    print('Saving camera matrix to: ', self.calibFileCam)
    print('Saving distortion coeffs to: ', self.calibFileDist)
    np.savetxt(self.calibFileCam, self.cameraMatrix)
    np.savetxt(self.calibFileDist, self.distCoeffs)
    print('Done.')
    return True


class CoordinateFrame(object):
  def __init__(self, axisLength):
    self.coordFrame = np.zeros((4, 3), np.float32)
    self.coordFrame[0, ...] = [0, 0, 0]
    self.coordFrame[1, ...] = [1, 0, 0]
    self.coordFrame[2, ...] = [0, 1, 0]
    self.coordFrame[3, ...] = [0, 0, -1]
    self.coordFrame *= axisLength
    

  def Draw(self, frame, rvec, tvec, calib):
    coords, _ = cv2.projectPoints(
        self.coordFrame, rvec, tvec, calib.cameraMatrix, calib.distCoeffs)
    # Draw coordinate axes.
    coords = [tuple(np.squeeze(x).astype(int)) for x in coords.tolist()]
    cv2.arrowedLine(frame, coords[0], coords[1], (  0, 255, 0), 2)
    cv2.arrowedLine(frame, coords[0], coords[2], (255, 255, 0), 2)
    cv2.arrowedLine(frame, coords[0], coords[3], (0, 255, 255), 2)



def RunPoseEstimation(video, outputDir, calib, chess):
  outputFile = os.path.join(
      outputDir,
      os.path.basename(video) + '-' + calib.Id() + '-detected_pose.mp4')
  camera = CameraSource(video, calib.imageHeight, outputFile)
  coordFrame = CoordinateFrame(chess.SquareWidth() * 2)

  rvec = None
  tvec = None
  useExtrinsicGuess = False

  while True:
    frame = camera.GetFrame()
    if frame is None:
      break
    objectPoints, imagePoints = chess.GetObjectAndImagePoints(
        frame, draw=False)
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

    if not camera.OutputFrameAndTestContinue('SolvePnP', frame, height=360)[0]:
      break


def main():
  #camera = 'pixel2'  
  camera = 'raspi'
  imageHeight = 720

  dataDir = '/Users/kwatra/Home/pvt/robotx/RobotX2020VisionSystem/data'
  calibDir = os.path.join(dataDir, 'calib_data')
  outputDir = os.path.join(dataDir, 'output')

  if camera == 'pixel2':
    calibVideo = os.path.join(calibDir, 'Chessboard-tv.mp4')
    maxSamples = 25
    squareWidth = 8.0
    rows = 6
    cols = 8
    startFrame = 0
  elif camera == 'raspi':
    calibVideo = os.path.join(calibDir, 'checkerboard-raspi.mov')
    maxSamples = 30
    squareWidth = 7.88
    rows = 6
    cols = 9
    startFrame = 65
  else:
    raise ValueError('Square width unkown for calib file: ', calibVideo)
  
  calib = Calibration(calibVideo,
                      imageHeight,
                      maxSamples,
                      startFrame=startFrame) 
  if not calib.LoadOrCompute(squareWidth, rows, cols):
    print('Could not load or compute calibration.')
    return
  else:
    print('Successfully loaded calibration.')

  chess = Chessboard(squareWidth, rows, cols)
  RunPoseEstimation(calibVideo, outputDir, calib, chess) 


if __name__ == '__main__':
  main()
