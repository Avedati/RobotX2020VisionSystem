import chessboard as cb
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from mpl_toolkits import mplot3d
from sklearn import linear_model


def DetectYellow(hsv):
    # Threshold the HSV image, keep only the color pixels
    mask = cv2.inRange(hsv, (25, 200, 100), (30, 255, 255))
    return np.atleast_3d(mask)

def DetectMagenta(hsv):
    # Threshold the HSV image, keep only the color pixels
    mask = cv2.inRange(hsv, (150, 200, 100), (170, 255, 255))
    return np.atleast_3d(mask)

def DetectOrange(hsv):
    # Threshold the HSV image, keep only the color pixels
    mask = cv2.inRange(hsv, (15, 200, 100), (20, 255, 255))
    return np.atleast_3d(mask)


def DrawProjectedSphere(center, radius, rvec, tvec, calib, frame, color, thickness=2):
  def world2cam(point):
    T = np.squeeze(tvec)
    R, _ = cv2.Rodrigues(rvec)
    return np.matmul(R, point) + T
  
  # Normalized vector from camera center to sphere center.
  axis = world2cam(center)
  axis_norm = np.linalg.norm(axis)
  if axis_norm <= radius:
    print('Sphere too close to camera')
    return
  axis = axis / axis_norm
  center = world2cam(center)

  # Projected circle will lie in plane perpendicular to axis. Decide upon an
  # arbitrary "x-axis" and "y-axis" in this plane.
  arbit_pt = np.array((1, 1, 1))
  arbit_pt[np.squeeze(np.argmax(axis))] = 0
  arbit_pt = arbit_pt / np.linalg.norm(arbit_pt)
  x_axis = np.cross(arbit_pt, axis)
  x_axis = x_axis / np.linalg.norm(x_axis)
  y_axis = np.cross(x_axis, axis)
  y_axis = y_axis / np.linalg.norm(y_axis)

  # Radius of projected circle:
  # r = r * cos(alpha)
  # where alpha is angle between axis (vector to center) and vector to tangent
  # to sphere. Hence sin(alpha) = R/L. cos(alpha) = sqrt(1-R^2/L^2)
  radius = radius * np.sqrt(1 - radius * radius / (axis_norm * axis_norm))

  # Sample points on circle.
  npts = 12
  points = []
  for i in range(npts):
    theta = i * 2 * np.pi / npts
    pt = center + radius * (x_axis * np.cos(theta) + y_axis * np.sin(theta))
    points.append(pt)
  points = np.asarray(points, np.float32)

  # Project to image plane. Use rvec and tvec of 0 since we already converted
  # to camera coordinates.
  points, _ = cv2.projectPoints(
        points, (0, 0, 0), (0, 0, 0), calib.cameraMatrix, calib.distCoeffs)

  # Fit and draw ellipse.
  box = cv2.fitEllipse(points)
  cv2.ellipse(frame, box, color, thickness)


class BallTracker(object):
  def __init__(self, calib):
    self.calib = calib
    self.image_center = tuple(self.calib.cameraMatrix[0:2, 2].astype(int))
    # Radius of ball
    self.r_ball = 3.5 # Inches.


  def Track(self, frame):
    colored_mask, gray, edges = self.GetBallEdgeImage(frame)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Only keep closed ones.
    #contours = [c for c,h in zip(contours, np.squeeze(hierarchy)) if h[2] >= 0 or h[3] < 0]

    #cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    mask_gray = np.max(colored_mask, axis=2) 
    mask = np.zeros(frame.shape[0:2], np.uint8)

    candidates = []
    for c in contours:
      if len(c) < 20:
        continue
      ellipse = cv2.fitEllipse(c)
      center, size, angle = ellipse
      mask = mask * 0
      cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
      mask_overlap = np.bitwise_and(mask, mask_gray)
      #if (cv2.countNonZero(mask_overlap) < 0.9 * np.pi * 0.25 * size[0] * size[1]):
      #  continue
      ratio = cv2.countNonZero(mask_overlap) / max(1, cv2.countNonZero(mask))
      if ratio < 0.9:
        continue

      cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
      box = cv2.boxPoints(ellipse).astype(int)
      cv2.drawContours(frame, [box], -1, (0, 255, 0), 1)

      center = (int(center[0]), int(center[1]))
      cv2.line(frame, self.image_center, center, (0, 0, 255), 1)

      # Compute 3D ball position.
      position = self.ComputeBallPosition(ellipse)

      DrawProjectedSphere(position,
                          self.r_ball,
                          (0, 0, 0),
                          (0, 0, 0),
                          self.calib,
                          frame,
                          (255, 0, 0),
                          1)


      candidates.append((position, c, ratio))

    # Pick candidate with largest ellipse/color-mask overlap : ellipse ratio.
    position = None
    contour = None
    if len(candidates) > 0:
      best = max(candidates, key=lambda c: c[2])
      position = best[0]
      contour = best[1]

    return position, contour, colored_mask, gray, edges


  def ComputeBallPosition(self, ellipse, use_major_axis=True):
    center, size, angle = ellipse

    # Axis from origin to center of ellipse. Same as what should be the major
    # axis of the ellipse.
    center = np.asarray(center)
    major_axis = center
    major_axis_norm = np.linalg.norm(major_axis)
    if major_axis_norm == 0:
      major_axis = (1, 0)
    else:
      major_axis = major_axis / major_axis_norm

    # Length of major axis from the detected ellipse.
    major_length = max(size)

    # Minor axis
    minor_length = min(size)
    minor_axis = np.array((major_axis[1], -major_axis[0]))

    # End points of major axis in the image.
    axis = major_axis if use_major_axis else minor_axis
    length = major_length if use_major_axis else minor_length
    ends = (center - axis * length / 2, center + axis * length / 2)
    ends = np.asarray(ends)
 
    # Normalized coordinates in image plane: correspond to a view frustum from
    # (-0.5, 0.5) and focal length of 1.
    ends = cv2.undistortPoints(ends, self.calib.cameraMatrix, self.calib.distCoeffs) 
    ends = np.squeeze(ends)

    # Normalized ellipse axis length in image plane is the diameter of
    # the projection.
    d_image = np.linalg.norm(ends[1] - ends[0])
    # Focal length is 1. Make it explicit so we still use the complete formula.
    f = 1.0 

    # The ends are 2D with the z-value implicitly 1. Convert to normalized 3D rays.
    rays = np.append(ends, [[1],[1]], axis=1)
    rays = rays / np.expand_dims(np.linalg.norm(rays, axis=1), -1)
    center_ray = rays[0,:] + rays[1,:]

    # Cosine angle of rays with z-axis == z-component of normalized rays.
    if use_major_axis:
      cosines = rays[:,2]
    else:
      # In theory, the minor axis projection is equivalent to rotating the camera
      # plane, such that x-axis aligns with major axis and the minor axis is the
      # projection of the sphere sitting in the x-z plane, hence the the center
      # ray should be considered as in x-z plane. So we need to compute cosine of
      # the rays w.r.t the center ray.
      # Does NOT seem to work in practice though, so something wrong in the
      # theory.
      center_ray /= np.linalg.norm(center_ray)
      cosines = np.dot(rays, center_ray)

    # z is obtained using the perspective formula on foreshortened image radius:
    # z = (R/r' * f), where r' = r*cos(alpha), where alpha is angle between ray
    # and z-axis. For ellipse, diameter is axis length and cos(alpha) is replaced
    # by the sum of inverse of fore-shortened radii obtained at the ends of
    # the axis.
    z = (self.r_ball / d_image) * f * (1.0/cosines[0] + 1.0/cosines[1])

    # Sphere center is along the bisector of *3D* rays (may be different from 2D
    # ellipse center), scaled to match the computed z value.
    sphere_center = center_ray * z / center_ray[2]

    return sphere_center
     

  def GetBallEdgeImage(self, frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = DetectYellow(hsv)
    magenta = DetectMagenta(hsv)
    orange = DetectOrange(hsv)
    colored_mask = (yellow * [0, 1, 1] +
                    magenta * [1, 0, 1] +
                    orange * [0, 0.5, 1]).astype(np.uint8)
  
    gaussian = cv2.GaussianBlur(colored_mask, (3, 3), 0)
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY) 
    kernel = np.ones((5, 5), np.uint8) 
  
    lower = 80
    upper = 160
    canny0 = cv2.Canny(gaussian[:,:,0], lower, upper)
    canny1 = cv2.Canny(gaussian[:,:,1], lower, upper)
    canny2 = cv2.Canny(gaussian[:,:,2], lower, upper)
    canny = np.maximum(canny0, np.maximum(canny1, canny2))
    canny = cv2.dilate(canny, kernel, iterations=1)
    canny = cv2.erode(canny, kernel, iterations=1)
    return colored_mask, gray, canny

    
  def PlotBallPositions(self, positions, outputFile=None, frame=None):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    frames = positions[:,0]
    xyz = positions[:,1:]
   
    # Split trajectories when more than a few frames are missing.
    boundaries = [0]
    for i in range(1, len(frames)):
      if frames[i-1] + 1 < frames[i] - 3:
        boundaries.append(i)
    boundaries.append(len(frames))

    print(frames)
    print(xyz)
    print(boundaries)

    normal, mask, a, b = self.FitPlane(xyz)

    for i in range(len(positions)):
      p0 = positions[i][1]
      p1 = positions[i][2]
      p2 = positions[i][3]
      # https://math.stackexchange.com/questions/1167717/transform-a-plane-to-the-xy-plane
      cos_theta = -1 / math.sqrt(a**2 + b**2 + 1)
      sin_theta = math.sqrt((a**2 + b**2) / (a**2 + b**2 + 1))
      u1 = b / math.sqrt(a**2 + b**2 + 1)
      u2 = -a / math.sqrt(a**2 + b**2 + 1)
      mat = [
        [cos_theta + (u1**2) * (1 - cos_theta), u1 * u2 * (1 - cos_theta), u2 * sin_theta],
        [u1 * u2 * (1 - cos_theta), cos_theta + (u2**2) * (1 - cos_theta), -u1 * sin_theta],
        [-u2 * sin_theta, u1 * sin_theta, cos_theta]
      ]
      positions[i][1] = p0 * mat[0][0] + p1 * mat[0][1] + p2 * mat[0][2]
      positions[i][2] = p0 * mat[1][0] + p1 * mat[1][1] + p2 * mat[1][2]
      positions[i][3] = p0 * mat[2][0] + p1 * mat[2][1] + p2 * mat[2][2]

    if outputFile is not None:
      np.savetxt(outputFile, positions)

    def plot_xyz(xyz, lines=True, points=True):
      # Swap y and z and negate z for better default visualization.
      x = xyz[:,0]
      y = xyz[:,2]
      z = -xyz[:,1]
      if lines:
        ax.plot3D(x, y, z)
      if points:
        ax.scatter3D(x, y, z, c=z, cmap='Greens'); 

    for i in range(1, len(boundaries)):
      b0 = boundaries[i - 1]
      b1 = boundaries[i]
      print(b0, b1)
      # Plot current contiguous segment.
      plot_xyz(xyz[b0:b1,:], lines=False)

    xyz_in = xyz[mask, :]
    plot_xyz(xyz_in)

    pt0 = xyz_in[0,:]
    pt1 = pt0 + normal * 20
    points = np.asarray([pt0, pt1])
    plot_xyz(points)

    """
    points, _ = cv2.projectPoints(
        points, (0, 0, 0), (0, 0, 0), self.calib.cameraMatrix, self.calib.distCoeffs)

    points = np.squeeze(points).astype(int)
    print(points)
    print(tuple(points[0]))
    print(tuple(points[1]))
    cv2.line(frame, tuple(points[0]), tuple(points[1]), (255, 0, 0), 2)
    ret, key = cb.ShowFrameAndTestContinue('Normal', frame)
    """

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.show()


  def FitPlane(self, xyz):
    """
    Computes parameters for a local regression plane using RANSAC
    https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
    """
    # Fit Z = aX + bY + d
    XY = xyz[:,:2]
    Z  = xyz[:,2]
    ransac = linear_model.RANSACRegressor(
        linear_model.LinearRegression(), residual_threshold=2)
    ransac.fit(XY, Z)

    inlier_mask = ransac.inlier_mask_
    coeff = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_

    # Plane normal.
    # Z = aX + bY + d
    # aX + bY -Z + d = 0
    # Normal: (a, b, -1)
    normal = np.asarray(tuple(coeff) + (-1,))
    a = normal[0]
    b = normal[1]
    normal /= np.linalg.norm(normal)

    print(normal)
    print(intercept)
    print(inlier_mask)

    return normal, inlier_mask, a, b



def main():
  camera = 'pixel2'  
  #camera = 'raspi'
  imageHeight = 720

  device = 'laptop_abhi'

  if device == 'laptop_kwatra':
    dataDir = '/Users/kwatra/Home/pvt/robotx/RobotX2020VisionSystem/data'
  elif device == 'laptop_abhi':
    dataDir = '/Users/spiderfencer/RobotX2020VisionSystem/data'

  calibDir = os.path.join(dataDir, 'calib_data')
  inputDir = os.path.join(dataDir, 'ball_videos')
  outputDir = os.path.join(dataDir, 'output')

  if camera == 'pixel2':
    #videoSource = os.path.join(inputDir, 'ball-sim-pixel2-sphere4.mp4')
    videoSource = os.path.join(inputDir, 'frcball-1.mp4')
  elif camera == 'raspi':
    return
    #videoSource = os.path.join(inputDir, 'vision-tape-raspi-2.mov')
  else:
    raise ValueError('Unknown camera type.')

  if camera == 'pixel2':
    calibVideo = os.path.join(calibDir, 'chessboard-tv.mp4')
    maxSamples = 25
  elif camera == 'raspi':
    calibVideo = os.path.join(calibDir, 'checkerboard-raspi.mov')
    maxSamples = 30
  else:
    raise ValueError('Unknown camera type.')

  calib = cb.Calibration(calibVideo, 720, maxSamples)
  if not calib.LoadOrCompute(finalImageHeight=imageHeight):
    print('Could not load or compute calibration.')
    return

  outputVideo = os.path.join(outputDir,
      os.path.basename(videoSource) + '-' + calib.Id() + '-ball-tracked.mp4')
  outputTracks = os.path.join(outputDir,
      os.path.basename(videoSource) + '-' + calib.Id() + '-ball-tracks.txt')

  camera = cb.CameraSource(videoSource, calib.imageHeight, outputVideo)

  tracker = BallTracker(calib)

  positions = []
  nFrames = 0
  while True:
    frame = camera.GetFrame()
    if frame is None:
      break
    
    position, contour, colored_mask, gray, edges = tracker.Track(frame)

    if position is not None:
      positions.append(np.array((nFrames,) + tuple(position)))
 
    #ret, key = cb.ShowFrameAndTestContinue('Output', frame)
    ret, key = camera.OutputFrameAndTestContinue('Output', frame)
    if not ret:
      break;
    
    nFrames += 1

  tracker.PlotBallPositions(np.asarray(positions), outputFile=outputTracks, frame=frame)
   


if __name__ == '__main__':
  main()
