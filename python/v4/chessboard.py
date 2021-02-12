import cv2
import math
import numpy as np
import os
import time

from videocaptureasync import VideoCaptureAsync
"""
from cscore import CameraServer
"""

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
  def __init__(self, cameraSource, height, output_file=None, startFrame=0,
               async_read=False, outputToServer=False, capture_size=None):
    if async_read:
      self.camera = VideoCaptureAsync(cameraSource)
    else:
      self.camera = cv2.VideoCapture(cameraSource)

    if capture_size is not None:
      print(capture_size)
      self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_size[0])
      self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])

    if async_read:
      self.ORIGINAL_WIDTH = self.camera.width
      self.ORIGINAL_HEIGHT = self.camera.height
    else:
      self.ORIGINAL_WIDTH = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
      self.ORIGINAL_HEIGHT = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('CameraSource')
    print('Requested capture size', capture_size)
    print('Actual capture size', self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT)

    self.HEIGHT = height
    self.WIDTH = self.ORIGINAL_WIDTH * self.HEIGHT // self.ORIGINAL_HEIGHT
    self.WIDTH = self.WIDTH + self.WIDTH % 2  # Make it even.

    self.startFrame = startFrame
    self.nFrames = 0
    self.writer = VideoWriter(output_file)

    if async_read:
      self.camera.start()

    self.outputToServer = outputToServer
    if outputToServer:
      # https://robotpy.readthedocs.io/en/stable/vision/code.html
      pass
      #self.outputStream = CameraServer.getInstance().putVideo(
      #    'ProcessedVisionFrame', self.WIDTH, self.HEIGHT)


  def GetFrame(self):  
    # Processing on first call. 
    if self.nFrames == 0:
      # Skip some frames if requested.
      if self.startFrame > 0:
        skippedFrames = 0
        while True:
          ret, frame = self.camera.read()
          if not ret or frame is None:
            print('No more frames')
            return None
          skippedFrames += 1
          if skippedFrames >= self.startFrame:
            break
      # Start timer for first frame.
      self.startTime = time.time()

    # Get frame.
    frame = None
    frameTime = time.time() 
    if self.nFrames > 0 and self.nFrames % 50 == 0:
      print('FPS: ', self.nFrames / (frameTime - self.startTime))
    self.nFrames += 1
    ret, frame = self.camera.read()
    if ret and frame is not None:
      if frame.shape[0] != self.HEIGHT:
        frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
    return frame


  def ImageSize(self):
    return (self.WIDTH, self.HEIGHT)


  def OutputFrameAndTestContinue(self, message, frame, height=None):
    self.writer.OutputFrame(frame)
    if self.outputToServer:
      self.outputStream.putFrame(frame)
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
    self.calibHeight = imageHeight
    self.imageHeight = imageHeight
    self.calibVideo = calibVideo
    self.maxSamples = maxSamples
    self.startFrame = startFrame
    name = self.Id()
    self.calibFileCam = calibVideo + '-' + str(name) + '-calib_cam.txt'
    self.calibFileDist = calibVideo + '-' + str(name) + '-calib_dist.txt'
    self.calibFileSize = calibVideo + '-calib_size.txt'
    self.hasCalib = False
    self.calibVideoSize = None
    self.cameraMatrix = None
    self.distCoeffs = None


  def Id(self):
    return str(self.imageHeight) + '-' + str(self.maxSamples)


  def ImageHeight(self):
    return self.imageHeight

    
  def ImageWidth(self):
    imageWidth = self.imageHeight * self.calibVideoSize[0] // self.calibVideoSize[1]
    imageWidth = imageWidth + imageWidth % 2  # Make it even.
    return imageWidth
    

  def PrintInfo(self):
    print('Calibration files:')
    print(self.calibFileSize)
    print(self.calibFileCam)
    print(self.calibFileDist)
    print('Calib original image height', self.calibHeight)
    print('Calib final image height', self.imageHeight)
    print('Calib final image width', self.ImageWidth())
    print('Calib video size', self.calibVideoSize)
    if self.hasCalib:
      print('CameraMatrix:\n', self.cameraMatrix)
      print('DistCoeffs:\n', self.distCoeffs)
    else:
      print('Calibration not loaded or computed.')


  def LoadFromFile(self, finalImageHeight):
    if self.hasCalib:
      return True
    
    # Read from file.
    self.hasCalib = False
    if (os.path.exists(self.calibFileSize) and
        os.path.exists(self.calibFileCam) and
        os.path.exists(self.calibFileDist)):
      print('Loading calibration info from files.')
      self.calibVideoSize = np.loadtxt(self.calibFileSize).astype(int)
      success = self.calibVideoSize.shape == (2,)
      if success:
        self.cameraMatrix = np.loadtxt(self.calibFileCam)
        success = self.cameraMatrix.shape == (3, 3)
        if success:
          self.distCoeffs = np.loadtxt(self.calibFileDist)
          success = self.distCoeffs.shape == (5,)
          if success:
            self.hasCalib = True
            print('Loaded successfully.')
            self.PrintInfo()
            self._RecomputeForNewSize(finalImageHeight)
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
                    forceRecompute=False,
                    finalImageHeight=None):
    """Loads or computes calibration.
       When finalImageHeight is given, scales camera matrix to final height,
       which may be different from the imageHeight at which calibration is
       computed.
    """
    if forceRecompute:
      print('Forcing recomputation of calibration data.')
    elif self.LoadFromFile(finalImageHeight):
      return True

    if squareWidth == None or rows == None or cols == None:
      raise ValueError(
          'Need to pass chessboard params to compute calibration')

    nChessFrames = 0
    objectPoints = []
    imagePoints = []
 
    camera = CameraSource(
        self.calibVideo, self.calibHeight, startFrame=self.startFrame)
    self.calibVideoSize = (camera.ORIGINAL_WIDTH, camera.ORIGINAL_HEIGHT)

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
    print('Saving calib video size to: ', self.calibFileSize)
    print('Saving camera matrix to: ', self.calibFileCam)
    print('Saving distortion coeffs to: ', self.calibFileDist)
    np.savetxt(self.calibFileSize, self.calibVideoSize)
    np.savetxt(self.calibFileCam, self.cameraMatrix)
    np.savetxt(self.calibFileDist, self.distCoeffs)
    print('Done.')

    self._RecomputeForNewSize(finalImageHeight)

    return True


  def _RecomputeForNewSize(self, imageHeight):
    if imageHeight is None or not self.hasCalib:
      return
    if imageHeight == self.imageHeight:
      return
    # Multiply focal length and center by height ratio.
    # All other entries are 0 or 1 (bottom-right, so keep that at 1).
    self.cameraMatrix *= imageHeight / self.imageHeight
    self.cameraMatrix[2, 2] = 1.0
    self.imageHeight = imageHeight
    print('Updated calibration to new height', imageHeight)
    self.PrintInfo()



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
  camera = CameraSource(
      video, calib.imageHeight, outputFile, capture_size=calib.calibVideoSize)
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
  #camera = 'raspi'
  camera = 'logitech'
  imageHeight = 720

  #dataDir = '/home/pi/RobotX2020VisionSystem/data'
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
  elif camera == 'logitech':
    calibVideo = os.path.join(calibDir, 'calib-logitech.mov')
    maxSamples = 30
    squareWidth = 8.5
    rows = 6
    cols = 9
    startFrame = 0
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
