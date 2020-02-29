import chessboard as cb
import cv2
import math
import numpy as np
import os
import time
#from tests import *
from networktables import NetworkTables

# https://robotpy.readthedocs.io/en/stable/guide/nt.html#client-initialization-driver-station-coprocessor
"""
import threading
from networktables import NetworkTables

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

NetworkTables.initialize(server='10.xx.xx.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

"""

sd = NetworkTables.getTable('SmartDashboard')

# Assume indexed contour for all functions here.
def max_x(contour):
  return max(contour, key=lambda ic: ic[1][0])
def max_y(contour):
  return max(contour, key=lambda ic: ic[1][1])
def min_x(contour):
  return min(contour, key=lambda ic: ic[1][0])
def min_y(contour):
  return min(contour, key=lambda ic: ic[1][1])
def nearest(pred, contour):
  return min(contour, key=lambda ic: np.linalg.norm(ic[1]-pred))
def min_cos(cosines, contour):
  return min(contour, key=lambda ic: cosines[ic[0]])


def canonical_contour(contour):
  # Point 0. Min-x value.
  idx0, cnt0 = min_x(contour)

  # Reset starting point of contour to leftmost.
  npts = len(contour)
  contour = sorted([((i - idx0 + npts) % npts, c) for i,c in contour])
  idx0 = 0

  # We want contour to be anti-clockwise. Reverse if not the case.
  # Check by looking at order of extremal y-points w.r.t (and excluding)
  # leftmost point.
  # This test may not always be right, but should work for our objects
  # of interest.
  idx_ylo, _ = min_y(contour[1:])
  idx_yhi, _ = max_y(contour[1:])
  clockwise = idx_ylo < idx_yhi
  if clockwise:
    contour = sorted([((npts - i) % npts, c) for i,c in contour])
  return contour


def merge_nearby_points(contour, min_dist):
  merged = []
  for ic in contour:
    if len(merged) == 0 or np.linalg.norm(merged[-1] - ic[1]) >= min_dist:   
      merged.append(ic[1])
  return list(enumerate(merged))


def contour_cosines(contour):
  points = [c for _,c in contour]
  points2 = points[1:] + points[:1]   # Next, with rotation.
  points0 = points[-1:] + points[:-1] # Prev, with rotation.
  cosines = []
  for p0, p, p2 in zip(points0, points, points2):
    v01 = p - p0
    v12 = p2 - p
    norm = np.linalg.norm(v01) * np.linalg.norm(v12)
    cosine = 1 if norm == 0 else np.dot(v01, v12) / norm
    cosines.append(cosine)
  return cosines


def select(contour, x_lo=None, y_lo=None, x_hi=None, y_hi=None, frame=None):
  subset = contour
  if x_lo is not None:
    subset = [(i, c) for i, c in subset if c[0] > x_lo]
  if y_lo is not None:
    subset = [(i, c) for i, c in subset if c[1] > y_lo]
  if x_hi is not None:
    subset = [(i, c) for i, c in subset if c[0] < x_hi]
  if y_hi is not None:
    subset = [(i, c) for i, c in subset if c[1] < y_hi]
  if frame is not None:
    x_lo = int(x_lo if x_lo is not None else 0)
    y_lo = int(y_lo if y_lo is not None else 0)
    x_hi = int(x_hi if x_hi is not None else frame.shape[1])
    y_hi = int(y_hi if y_hi is not None else frame.shape[0])
    cv2.rectangle(frame, (x_lo, y_lo), (x_hi, y_hi), (0, 255, 0))
  if len(subset) == 0:
    return [(-1, np.array((0, 0)))]
  return subset


def DrawPolygon(polygon,
                frame,
                line_color,
                circle_color=None,
                circle_radius=5,
                circle_thickness=2,
                draw_index=False,
                draw_index_scale=1):
  polygon = [np.squeeze(p) for p in list(polygon)]
  npts = len(polygon)
  for i in range(npts):
    x1 = int(polygon[i][0])
    y1 = int(polygon[i][1])
    x2 = int(polygon[(i+1)%npts][0])
    y2 = int(polygon[(i+1)%npts][1])
    cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
    if circle_color is not None:
      cv2.circle(frame, (x1, y1), circle_radius, circle_color, circle_thickness)
    if draw_index_scale > 0:
      cv2.putText(frame, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                  0.5 * draw_index_scale, (0, 0, 0))
  

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


def DrawProjectedCircleInXYPlane(center, radius, rvec, tvec, calib, frame, color, thickness=2):
  # Sample points on circle.
  npts = 12
  points = []
  for i in range(npts):
    theta = i * 2 * np.pi / npts
    points.append(
        center + np.array((radius * np.cos(theta), radius * np.sin(theta), 0)))
  points = np.asarray(points, np.float32)

  # Project to image plane.
  points, _ = cv2.projectPoints(
        points, rvec, tvec, calib.cameraMatrix, calib.distCoeffs)

  # Fit and draw ellipse.
  box = cv2.fitEllipse(points)
  cv2.ellipse(frame, box, color, thickness)
  #DrawPolygon(points, frame, color)



class TargetObject(object):
  def __init__(self,
               shape_match_threshold,
               min_area_ratio,
               max_area_ratio,
               min_aspect_ratio,
               max_aspect_ratio,
               scale):
    self.shape_matching=False
    self.shape_match_threshold = shape_match_threshold
    self.min_area_ratio = min_area_ratio
    self.max_area_ratio = max_area_ratio
    self.min_aspect_ratio = min_aspect_ratio
    self.max_aspect_ratio = max_aspect_ratio
    self.scale = scale
    # Inner radius of hexagon.
    self.r_in = 17.32
    # Outer radius: Tap is 2in wide. outer radius adds diagonal with 60-degrees.
    self.r_out = self.r_in + 2.0 / np.sin(np.pi/3)
    # Back circle (z-offset and radius)
    self.r_back = 6.49
    self.z_back = 29.25
    # Height of hexagon center.
    self.y_base = 98.25
 

  def GetContourMatches(self, contours):
    matches = []
    for c in contours:
      # Area matching convex hull.
      hull_area = cv2.contourArea(cv2.convexHull(c))
      if hull_area < 200:
        continue
      area_ratio = cv2.contourArea(c) / hull_area
      if area_ratio < self.min_area_ratio:
        continue
      if area_ratio > self.max_area_ratio:
        continue

      # Aspect ratio.
      r = cv2.boundingRect(c)
      aspect_ratio = r[2] / r[3]
      if aspect_ratio < self.min_aspect_ratio:
        continue
      if aspect_ratio > self.max_aspect_ratio:
        continue

      # Shape matching.
      if self.shape_matching:
        distance = cv2.matchShapes(self.points2d, c, cv2.CONTOURS_MATCH_I3, 0) 
        print('shape distance, threshold', distance, self.shape_match_threshold)
        if distance >= self.shape_match_threshold:
          continue

      # Unfiltered matches.
      metrics = {
          'area_ratio': area_ratio,
          'aspect_ratio': aspect_ratio,
      }
      if self.shape_matching:
        metrics.update({'shape_dist': distance})
      matches.append((c, metrics))

    return matches


  def DrawMatches(self, frame, matches, color, draw_metrics=False):
    if matches is None:
      return
    max_match_dist = self.shape_match_threshold
    for m in matches:
      cv2.drawContours(frame, [m[0]], -1, color, 2)
      if draw_metrics:
        point = tuple(np.squeeze(m[0][0]))
        metrics = str(int(m[1]['area_ratio']*100) / 100)
        if self.shape_matching:
          metrics += ' ' + str(int(m[1]['shape_dist'] * 100) / 100)
        cv2.putText(frame, metrics, tuple(point), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color)


  def GetBestMatch(self, contours, gray, frame=None, color=None):
    contour_metrics = self.GetContourMatches(contours)
    matches = []
    for c,m in contour_metrics:
      c = self.FindCorners(c, frame=None)
      if c is not None and self.TestCorners(c):
        matches.append((c, m))
   
    if len(matches) == 0:
      return None

    matches.sort(key=lambda cm: -cv2.arcLength(cm[0], True))

    best_match = (matches[0][0]).astype(np.float32)

    # Corener sub-pix.
    window = 2 * int(2 * self.scale) + 1
    best_match = cv2.cornerSubPix(
        gray,
        best_match,
        (window, window),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))

    # Only keep bottom contour.
    img_pts = list(best_match)[0:]
    obj_pts = list(self.points3d)[0:]

    if frame is not None:
      # Draw all matches in overlay.
      overlay_color = tuple([255-c for c in color])
      alpha = 0.4
      overlay = frame.copy()
      self.DrawMatches(overlay, matches, overlay_color, draw_metrics=True)
      cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

      # Draw best match in opposite color.
      DrawPolygon(img_pts, frame, line_color=color, circle_color=overlay_color,
                  draw_index=False, draw_index_scale=self.scale)
    
    return img_pts, obj_pts



class TapeObject(TargetObject):
  def __init__(self,
               shape_match_threshold=1.2,
               min_area_ratio=0.1,
               max_area_ratio=0.4,
               min_aspect_ratio=1.0,
               max_aspect_ratio=4.0,
               **kwargs):
    super().__init__(shape_match_threshold,
                     min_area_ratio,
                     max_area_ratio,
                     min_aspect_ratio,
                     max_aspect_ratio,
                     **kwargs)
    self.points3d = np.zeros((8, 3), np.float32)
    for i in range(4):
      theta = np.pi + i * np.pi / 3
      self.points3d[7-i, 0:2] = (self.r_in * np.cos(theta), -self.r_in * np.sin(theta))
      self.points3d[  i, 0:2] = (self.r_out * np.cos(theta), -self.r_out * np.sin(theta))
    self.points2d = self.points3d[..., 0:2]


  def FindCorners(self, contour, frame=None):
    # Contour is now an indexed contour: Each element is (i, np.array(x, y)).
    contour = [ic for ic in enumerate(list(np.squeeze(contour)))]
    contour = canonical_contour(contour)
    contour = merge_nearby_points(contour, 5)
    cosines = contour_cosines(contour)

    # Contour points and indices. Init to zeros.
    cnt = [(0,0)] * 8

    # Point 0. First in canonical contour.
    _, cnt[0] = contour[0]

    # Point 3. Max-x value.
    idx3, cnt[3] = max_x(contour)

    # Mid point for left and right half planes.
    x_mid = (cnt[3][0] + cnt[0][0]) * 0.5
  
    # Approximage tape width in image space.
    obj = list(self.points2d)
    cnt03 = cnt[3] - cnt[0]
    norm03 = np.linalg.norm(cnt03)
    if norm03 == 0:
      return None
    obj2cnt = norm03 / np.linalg.norm(obj[3] - obj[0])
    tape_width = (self.r_out - self.r_in) * obj2cnt

    # Top and bottom contours excluding points 0 and 3.
    top_contour = contour[idx3+1:]
    bottom_contour = contour[1:idx3]

    # Point 1 and 2: left and right half, sharpest corners in bottom contour.
    _, cnt[1] = min_cos(cosines, select(bottom_contour, x_hi=x_mid, frame=frame))
    _, cnt[2] = min_cos(cosines, select(bottom_contour, x_lo=x_mid, frame=frame))

    # Point 4 and 7: extrapolate from 0 and 3, and pick nearest in left and
    # right half, in a flat band around the 0-3 line.
    pred4 = cnt[3] - cnt03 * tape_width / norm03
    pred7 = cnt[0] + cnt03 * tape_width / norm03
    y_lo = min(cnt[0][1], cnt[3][1]) - tape_width * 0.5
    y_hi = max(cnt[0][1], cnt[3][1]) + tape_width * 0.5
    _, cnt[4] = nearest(pred4, select(
        top_contour, x_lo=x_mid, y_lo=y_lo, y_hi=y_hi, frame=frame))
    _, cnt[7] = nearest(pred7, select(
        top_contour, x_hi=x_mid, y_lo=y_lo, y_hi=y_hi, frame=frame))

    # Point 5 and 6: extrapolate from 4 and 7, and pick nearest in left and right half.
    cnt32 = cnt[2] - cnt[3]
    cnt01 = cnt[1] - cnt[0]
    norm32 = np.linalg.norm(cnt32)
    norm01 = np.linalg.norm(cnt01)
    if norm01 == 0 or norm32 == 0:
      return None
    pred5 = cnt[4] + cnt32 * (1 - tape_width / norm32)
    pred6 = cnt[7] + cnt01 * (1 - tape_width / norm01)
    _, cnt[5] = nearest(pred5, select(top_contour, x_lo=x_mid, frame=frame))
    _, cnt[6] = nearest(pred6, select(top_contour, x_hi=x_mid, frame=frame))

    # If any prediction failed, return None.
    for c in cnt:
      if c[0] == 0 or c[1] == 0:
        return None

    if frame is not None:
      colors = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
      colors = colors + colors[::-1]
      for i,c in enumerate(cnt):
        cv2.circle(frame, tuple(c), 5, colors[i])
        cv2.putText(frame, str(i), tuple(c), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, (0, 0, 0))

    return np.asarray(cnt)


  def TestCorners(self, contour):
    corners = list(contour)
    if len(corners) != 8:
      return False
    
    # Bottom contour. x-values.
    for i in range(3):
      if corners[i+1][0] <= corners[i][0]:
        return False
    # Top contour. x-values.
    for i in range(3,8):
      if corners[(i+1)%8][0] >= corners[i][0]:
        return False
    # Test against bottom contour, bottom two points. y-values.
    for i in [0,3,4,6,7]:
      if corners[i][1] >= corners[1][1]:
        return False
    for i in [0,3,4,5,7]:
      if corners[i][1] >= corners[2][1]:
        return False
    # Test against top contour, bottom two points. y-values.
    for i in [0,3,4,7]:
      if corners[i][1] >= corners[5][1]:
        return False
      if corners[i][1] >= corners[6][1]:
        return False
    
    return True


class HexagonObject(TargetObject):
  def __init__(self,
               shape_match_threshold=1.2,
               min_area_ratio=0.99,
               max_area_ratio=1.01,
               min_aspect_ratio=0.25,
               max_aspect_ratio=4.0,
               **kwargs):
    super().__init__(shape_match_threshold,
                     min_area_ratio,
                     max_area_ratio,
                     min_aspect_ratio,
                     max_aspect_ratio,
                     **kwargs)
    self.points3d = np.zeros((6, 3), np.float32)
    for i in range(6):
      theta = np.pi + i * np.pi / 3
      self.points3d[i, 0:2] = (self.r_in * np.cos(theta), -self.r_in * np.sin(theta))
    self.points2d = self.points3d[..., 0:2]


  def FindCorners(self, contour, frame=None):
    # Contour is now an indexed contour: Each element is (i, np.array(x, y)).
    contour = [ic for ic in enumerate(list(np.squeeze(contour)))]
    contour = canonical_contour(contour)
    cosines = contour_cosines(contour)

    # Contour points and indices. Init to zeros.
    cnt = [(0,0)] * 6

    # Point 0. First in canonical contour.
    _, cnt[0] = contour[0]

    # Point 3. Max-x value.
    idx3, cnt[3] = max_x(contour)

    # Mid point for left and right half planes.
    x_mid = (cnt[3][0] + cnt[0][0]) * 0.5
  
    # Top and bottom contours excluding points 0 and 3.
    top_contour = contour[idx3+1:]
    bottom_contour = contour[1:idx3]

    # Point 1 and 2: left and right half, sharpest corners in bottom contour.
    _, cnt[1] = min_cos(cosines, select(bottom_contour, x_hi=x_mid, frame=frame))
    _, cnt[2] = min_cos(cosines, select(bottom_contour, x_lo=x_mid, frame=frame))

    # Point 4 and 5: left and right half, sharpest corners in top contour.
    _, cnt[4] = min_cos(cosines, select(top_contour, x_lo=x_mid, frame=frame))
    _, cnt[5] = min_cos(cosines, select(top_contour, x_hi=x_mid, frame=frame))

    # If any prediction failed, return None.
    for c in cnt:
      if c[0] == 0 or c[1] == 0:
        return None

    if frame is not None:
      colors = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
      colors = colors + colors[1:3]
      for i,c in enumerate(cnt):
        cv2.circle(frame, tuple(c), 5, colors[i])
        cv2.putText(frame, str(i), tuple(c), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, (0, 0, 0))

    return np.asarray(cnt)


  def TestCorners(self, contour):
    corners = list(contour)
    if len(corners) != 6:
      return False
    
    # Bottom contour. x-values.
    for i in range(3):
      if corners[i+1][0] <= corners[i][0]:
        return False
    # Top contour. x-values.
    for i in range(3,6):
      if corners[(i+1)%6][0] >= corners[i][0]:
        return False
    # Test against bottom contour, bottom two points. y-values.
    for i in [0] + [*range(3,6)]:
      if corners[i][1] >= corners[1][1]:
        return False
      if corners[i][1] >= corners[2][1]:
        return False
    # Test against top contour, top two points. y-values.
    for i in range(0,4):
      if corners[i][1] <= corners[4][1]:
        return False
      if corners[i][1] <= corners[5][1]:
        return False
    
    return True



class ObjectTracker(object):
  def __init__(self, calib):
    self.scale = calib.imageHeight / 360.0
    self.tape = TapeObject(scale=self.scale)
    self.hexagon = HexagonObject(scale=self.scale)
    self.calib = calib
    self.rvec = None
    self.tvec = None
    self.trackingSuccess = False
    self.cam_origin = None
    self.cam_orient = None
    self.rob_origin = None
    self.rob_orient = None
    self.launch_data = None
    
    # Radius of ball
    self.r_ball = 3.5 # Inches.
    
    # The vector between camera and back center should lie within a circle
    # around the front center, such that it is far enough away from target's
    # edges to allow the ball to pass.
    hexagon_shortest_dist = self.hexagon.r_in * np.sin(np.pi/3)
    eps = 1.0
    self.r_clear_front = hexagon_shortest_dist - self.r_ball - eps
    
    # Similar computation for back circle but with lower epsilon (we are OK if
    # it hits the edge of the back circle), because it would still have fallen
    # inside the hexagon in that case.
    eps = 0.1
    self.r_clear_back = self.hexagon.r_back - self.r_ball - eps

    # Robot configuration.
    # TODO: Some of these should be computed as calibration between camera and
    # robot.
    self.robot_pitch = 42.0  # In degrees.
    # Yaw (in radians): angle from camera orientation to robot orientation.
    # Should be +ve if robot is to the left of camera.
    self.camera_to_robot_yaw = 0 #-10 * np.pi / 180
    # Offset vector: vector from camera origin to robot origin.
    self.camera_to_robot_vec = (0, 10, 5)  # (20, 10, 5)


  def Track(self, frame):
    gray, edges = self.GetEdgeImage(frame)
    self.TrackFromEdges(edges, gray, frame)
    return gray, edges


  def GetEdgeImage(self, frame):
    gaussian = cv2.GaussianBlur(frame, (7, 7), 0)
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
    return gray, canny


  def TrackFromEdges(self, edges, gray, frame):
    # Extract contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Only keep closed ones.
    #contours = [c for c,h in zip(contours, np.squeeze(hierarchy)) if h[2] >= 0 or h[3] < 0]
    
    # Convert to polygons.
    polygons = [self.ExtractPolygon(c) for c in contours]
    polygons = [c for c in polygons if len(c) < 20]
    polygons = [c for c in polygons if cv2.arcLength(c, False) > 200 * self.scale]

    img_pts = []
    obj_pts = []
    tape_pts = self.tape.GetBestMatch(polygons, gray, frame, (0, 0, 255))
    if tape_pts is not None:
      # Only keep bottom contour from tape for estimation.
      img_pts = tape_pts[0][:4]
      obj_pts = tape_pts[1][:4]
    else:
      hex_pts = self.hexagon.GetBestMatch(polygons, gray, frame, (0, 255, 255))
      if hex_pts is not None:
        img_pts = hex_pts[0]
        obj_pts = hex_pts[1]

    self.RunPoseEstimationTarget(img_pts, obj_pts, frame)
    self.ComputeCameraAndRobotState()
    self.ComputeLaunchData(frame)
    
    return


  def ExtractPolygon(self, contour):
    # Arc-length based eps seems to work better on vision tape.
    eps = 0.005 * cv2.arcLength(contour, True)
    #eps = 5 #3
    polygon = cv2.approxPolyDP(contour, eps, True)
    return polygon


  def RunPoseEstimationTarget(self, img_pts, obj_pts, frame=None):
    success = len(img_pts) >= 4
    if success:
      img_pts = np.asarray(img_pts)
      obj_pts = np.asarray(obj_pts)
      success, rvec, tvec = cv2.solvePnP(
          obj_pts, img_pts, self.calib.cameraMatrix, self.calib.distCoeffs,
          rvec=self.rvec, tvec=self.tvec, useExtrinsicGuess=self.trackingSuccess)

    self.trackingSuccess = success
    if success:
      self.rvec = rvec
      self.tvec = tvec
    
    # Draw projection of original targets.
    if frame is not None and self.rvec is not None and self.tvec is not None:
      # Draw back circle.
      DrawProjectedCircleInXYPlane((0.0, 0.0, self.hexagon.z_back),
                                   self.hexagon.r_back,
                                   self.rvec,
                                   self.tvec,
                                   self.calib,
                                   frame,
                                   (0, 0, 255))

      # Draw projections of hexagon target points.
      hexagon, _ = cv2.projectPoints(self.hexagon.points3d,
                                     self.rvec,
                                     self.tvec,
                                     self.calib.cameraMatrix,
                                     self.calib.distCoeffs)
      DrawPolygon(hexagon,
                  frame,
                  line_color=(255, 0, 0),
                  circle_color=(0, 127, 255),
                  circle_radius=6,
                  circle_thickness=3)

      # Draw projections of tape target points.
      tape, _ = cv2.projectPoints(self.tape.points3d,
                                  self.rvec,
                                  self.tvec,
                                  self.calib.cameraMatrix,
                                  self.calib.distCoeffs)
      DrawPolygon(tape,
                  frame,
                  line_color=(0, 255, 0),
                  circle_color=(0, 127, 255),
                  circle_radius=6,
                  circle_thickness=3)

      # Draw coordinate axes.
      coord_frame = cb.CoordinateFrame(self.hexagon.r_in * 0.5)
      coord_frame.Draw(frame, self.rvec, self.tvec, self.calib)


  def ComputeCameraAndRobotState(self):
    """Computes camera and robot origin and orientation in world coordinates.
    And target center in robot coordinates.
    """
    if self.rvec is None or self.tvec is None:
      return

    tvec = np.squeeze(self.tvec)
    rmat, _ = cv2.Rodrigues(self.rvec)

    def cam2world(point):
      return np.matmul(np.transpose(rmat), (point - tvec))

    # Camera origin and orientation (z-vector) in world coords.
    cam_origin = cam2world(np.zeros((3,)))
    cam_orient = cam2world(np.asarray([0, 0, 1])) - cam_origin

    # Robot orientation (z-vector) in world coordinates:
    # Project to x-z plane and apply camera to robot rotation about y-axis.
    x, _, z = tuple(cam_orient)
    yaw = self.camera_to_robot_yaw
    rob_z = np.array((x * np.cos(yaw) - z * np.sin(yaw), 0,
                      x * np.sin(yaw) + z * np.cos(yaw)))
    rob_norm = np.linalg.norm(rob_z)
    if rob_norm > 0:
      rob_z = rob_z / rob_norm
    # Robot y-vector is same as world-y. x-vector is normal to both.
    # rob_z:(x,0,z) maps to rob_x:(z,0,-x)
    rob_y = np.array((0, 1, 0))
    rob_x = np.array((rob_z[2], 0, -rob_z[0]))

    # Robot origin in world coords.
    rob_origin = cam_origin + self.camera_to_robot_vec

    # Front target (world origin == hexagon center) in robot coords.
    #
    # Same as projection of robot_to_target vector to robot coordinate system.
    target_front = np.array((0, 0, 0))
    rob_to_tgt_w = target_front - rob_origin
    rob_to_tgt_r = np.array((np.dot(rob_to_tgt_w, rob_x), rob_to_tgt_w[2],
                             np.dot(rob_to_tgt_w, rob_z)))

    # Update state.
    self.cam_origin = cam_origin
    self.cam_orient = cam_orient
    self.rob_origin = rob_origin
    self.rob_orient = rob_z
    self.target_in_robot_coords = rob_to_tgt_r


  def ComputeLaunchData(self, frame=None):
    if (self.rob_origin is None or self.rob_orient is None or
        self.rvec is None or self.tvec is None):
      return

    def compute_yaw(target):
      """Computes rotation (about y-axis) that would align robot with target.
         +ve angle => counter-clockwise => turn left.
         -ve angle => clockwise => turn right.
         Sign changed in network table to make -ve imply left and vice-versa.
      """
      # Vector from robot to target in world coords.
      rob_to_tgt = target - self.rob_origin

      # Project robot-target vector and robot orientation in x-z plane. 
      x1, z1 = (self.rob_orient[0], self.rob_orient[2]) 
      x2, z2 = (rob_to_tgt[0], rob_to_tgt[2]) 

      # https://math.stackexchange.com/questions/317874/calculate-the-angle-between-two-vectors
      # Let 𝑎=(𝑥1,𝑦1), 𝑏=(𝑥2,𝑦2). If 𝜃 is the "oriented" angle from 𝑎 to 𝑏
      # (that is, rotating 𝑎̂  by 𝜃 gives 𝑏̂ ), then: 𝜃=atan2(𝑥1𝑦2−𝑦1𝑥2,𝑥1𝑥2+𝑦1𝑦2)
      yaw = math.atan2(x1*z2 - z1*x2, x1*x2 + z1*z2)
      return yaw, rob_to_tgt

    #-------------------------------------------------------------------------#
    target_front = np.array((0, 0, 0))
    target_back = np.array((0, 0, self.hexagon.z_back))

    # Robot location and correction angles.
    yaw_front, rob_to_front = compute_yaw(target_front)
    yaw_back, rob_to_back = compute_yaw(target_back)

    # Launch data for 3 scenarios:
    # 0: Current orientation
    # 1: Orient towards front target (hexagon)
    # 2: Orient towards back target (circle)
    #
    #
    # Angles are alignment angles by which to turn the robot to aim for the
    # front and back targets in degrees.
    # Change sign of yaw angle, so left turn is -ve and right turn is +ve.
    self.launch_data = [{}, {}, {}]
    self.launch_data[0] = {'orient': self.rob_orient, 'angle': 0}
    self.launch_data[1] = {'orient': rob_to_front, 'angle': -yaw_front * 180/np.pi}
    self.launch_data[2] = {'orient': rob_to_back, 'angle': -yaw_back * 180/np.pi}

    # Determine the "ball plane" and viability for each of the three
    # orientations: current, towards-front-center, towards-back-center.
    # Ball plane: 3 points (base, aim_front, aim_back).
    # Viability: 2 booleans (front target viable, back target viable).
    # Speed: 2 floats (front target speed, back target speed) if viable.
    for i in range(3):
      points, viable, speeds = self.ComputeBallPlaneAndSpeed(
          self.launch_data[i]['orient'], self.rob_origin)
      self.launch_data[i].update(
          {'points': points, 'viable': viable, 'speeds': speeds})
    
    # Write to network tables.
    # TODO: Add speed and any other info.
    sd.putNumber('AngleFrontPort', self.launch_data[1]['angle'])
    sd.putNumber('AngleBackPort', self.launch_data[2]['angle'])
    sd.putBoolean('TrackingSuccess', self.trackingSuccess)

    if frame is not None:
      # Clearance circles.
      DrawProjectedCircleInXYPlane((0.0, 0.0, 0.0),
                                   self.r_clear_front,
                                   self.rvec,
                                   self.tvec,
                                   self.calib,
                                   frame,
                                   (255, 0, 255)) 
      DrawProjectedCircleInXYPlane((0.0, 0.0, self.hexagon.z_back),
                                   self.r_clear_back,
                                   self.rvec,
                                   self.tvec,
                                   self.calib,
                                   frame,
                                   (0, 127, 255)) 

      # Ball plane intersection with target plane.
      colors = [(255, 0, 0), (255, 0, 255), (0, 127, 255)]
      names = ['Current aim', '  Front aim', '   Back aim']
      for i in range(3):
        name = names[i] + ' '
        color = colors[i]
        data = self.launch_data[i]
        angle_text = name + ' ' + str(int(data['angle'] * 100) / 100)
        viable = data['viable']
        viable_text = [n + ('OK' if x else '__') for n,x in zip(['F ','B '], viable)]
        viable_color = [(0, 255, 0) if x else (0, 0, 255) for x in viable]
        cv2.putText(frame, angle_text, (10, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color)
        cv2.putText(frame, viable_text[0], (200, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    viable_color[0])
        cv2.putText(frame, viable_text[1], (260, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    viable_color[1])

        # Draw line from target "base" (same y-level as robot) to front aim.
        # Draw an 'x' for back aim location.
        points, _ = cv2.projectPoints(np.asarray(data['points']),
                                      self.rvec,
                                      self.tvec,
                                      self.calib.cameraMatrix,
                                      self.calib.distCoeffs) 
        points = [tuple(np.squeeze(x).astype(int)) for x in points.tolist()]
        points = [(min(x, frame.shape[1]), min(y, frame.shape[0])) for x,y in points] 
        cv2.arrowedLine(frame, points[0], points[1], color, 2, tipLength=0.01)
        cv2.drawMarker(frame, points[2], color, cv2.MARKER_TILTED_CROSS, 10, 2)


  def ComputeBallPlaneAndSpeed(self, robot_to_target, rob_origin):
    """Determine the "ball plane" within which the ball will fly towards the
    target, given the robot_to_target direction vector.

    Ball plane is the plane between the robot orientation and y-axis. Returned
    as three points:
      base: On the target's front plane at the same y-value as the robot 
      aim_front: On target's front plane's center at y-value of 0.
      aim_back: On target's back circle's center at y-value of 0. 
    Checks the viability:
      Tuple indicating if entering (front, back) target is viable.
    Ball speed:
      Tuple indicating ball speed for (front, back) target, if viable.
    """
    failure = (((0.0, 0.0, 0.0),)*3, (False, False), (0.0, 0.0))
    if not self.trackingSuccess:
      return failure

    # Normalized robot-target vector in x-z plane. Since we are computing
    # the ball plane assuming robot is oriented in this direction, this is
    # same as the z-vector in robot's coordinate system.
    rob_z = (robot_to_target[0], 0, robot_to_target[2])
    norm_z = np.linalg.norm(rob_z)
    if norm_z == 0:
      print('Encountered zero-norm robot orientation')
      return failure
    rob_z = rob_z / norm_z

    # Find point of intersection of ray from robot to z=0 (target's) plane.
    # Ray-plane intersection:
    # Find d s.t. the point (rob_origin + d * rob_z) satisfies plane
    # equation (z = 0).
    # equiv. to solving rob_origin[2] + d * rob_z[2] = 0.
    if abs(rob_z[2]) < 0.01:
      print('Robot oriented 90-degrees away from target plane.')
      return failure
    df = -rob_origin[2]/rob_z[2]
    base = rob_origin + df * rob_z
    aim_front = np.array((base[0], 0, base[2]))

    # For back plane solve for z=z_back.
    db = (self.hexagon.z_back - rob_origin[2])/rob_z[2]
    base_back = rob_origin + db * rob_z
    aim_back = np.array((base_back[0], 0, base_back[2]))

    # Launch is viable only if the x-location lies within clearance circle.
    # Else ball could hit an obstacle. Only x matters for initial viability,
    # since y location will depend on ball speed.
    viable_front = abs(aim_front[0]) < self.r_clear_front
    viable_back = viable_front and abs(aim_back[0]) < self.r_clear_back
   
    # TODO: Compute speeds.
    speeds = (0.0, 0.0)

    points = (base, aim_front, aim_back)
    viable = (viable_front, viable_back)

    return points, viable, speeds



class Ball(object):
  def __init__(self):
    self.g = -32.1740 * 12  # In inches / second^2
    self.fps = 30.0
    self.t = -1  # -ve value => Simulation has not started.

  
  def StartSimulation(self,
                      speed,
                      tracker,
                      launch_index, # 0:curr, 1:front, 2:back
                      max_secs=2.0,
                      max_z_from_back=50.0):
    # Check if we have valid robot orientation to start sim.
    launch_orient = None
    if tracker.launch_data is not None:
      if len(tracker.launch_data) == 3:
        launch_orient = tracker.launch_data[launch_index].get('orient')
    if launch_orient is None:
      print('Cannot start simulation. Launch orientation not known.')
      return

    # Sim state. In 2D space (z-y). z:horizontal, y:vertical.
    # Negative sin(theta) and g, since y points downward in our case.
    theta = tracker.robot_pitch * np.pi / 180
    self.vel = speed * np.array((np.cos(theta), -np.sin(theta)))
    self.pos = np.array((0.0, 0.0))
    self.t = 0
    self.entered_front = False
    self.entered_back = False

    # Mapping to 3D space.
    self.origin = tracker.rob_origin
    # Make sure horizontal orientation is in x-z plane.
    self.horz_dir = launch_orient * (1, 0, 1)
    self.horz_dir = self.horz_dir / max(0.0001, np.linalg.norm(self.horz_dir))
    self.vert_dir = np.array((0, 1, 0))

    # Sim params.
    self.t_max = max_secs
    self.max_z = tracker.hexagon.z_back + max_z_from_back
    self.step_tolerance = 0.1  # Inches. Desired max motion during sim steps.


  def Simulate(self, tracker, step_fps_frac=0.1):
    if self.t < 0:
      return
   
    def sim_step(pos, vel, dt):
      z = pos[0] + vel[0] * dt
      y = pos[1] + vel[1] * dt - 0.5 * self.g * dt * dt
      u = vel[0]
      v = vel[1] - self.g * dt
      return (z,y), (u,v)

    def update_entering_state(prev_pos, curr_pos):
      x0, y0, z0 = tuple(self.Position3D(prev_pos))
      x1, y1, z1 = tuple(self.Position3D(curr_pos))
      if not self.entered_front:
        if z0 < 0 and z1 >= 0:
          if np.linalg.norm((x1, y1)) <= tracker.r_clear_front:
            self.entered_front = True
      if self.entered_front and not self.entered_back:
        if z0 < tracker.hexagon.z_back and z1 >= tracker.hexagon.z_back:
          if np.linalg.norm((x1, y1)) <= tracker.r_clear_back:
            self.entered_back = True

    # Total time step covered in this call.
    total_step = 1.0 / self.fps * step_fps_frac

    # Size of intermediate time steps. Smaller than total, so we can detect
    # target plane corssings reliably. Only need this at crossings but doing
    # it for the entire simulation for simplicity.
    interim_step = total_step
    speed = np.linalg.norm(self.vel)
    if speed > 0:
      interim_step = min(total_step, self.step_tolerance/speed)

    dt = 0
    pos, vel = self.pos, self.vel
    while dt < total_step:
      prev_pos = pos
      pos, vel = sim_step(pos, vel, interim_step)
      update_entering_state(prev_pos, pos)
      dt += interim_step

    self.pos = pos
    self.vel = vel
    self.t += dt
    if self.t >= self.t_max or self.Position3D(self.pos)[2] > self.max_z:
      self.t = -1


  def Position3D(self, pos):
    return self.origin + pos[0] * self.horz_dir + pos[1] * self.vert_dir


  def Draw(self, tracker, frame):
    if self.t < 0 or tracker.rvec is None or tracker.tvec is None:
      return
 
    if self.entered_back:
      color = (0, 127, 255)
    elif self.entered_front:
      color = (255, 0, 255)
    else:
      color = (0, 255, 255)
    
    position = self.Position3D(self.pos)
    if position[2] > tracker.hexagon.z_back:
      intensity = 0.6
    elif position[2] > 0:
      intensity = 0.8
    else:
      intensity = 1.0
    color = tuple([int(x * intensity) for x in color])
      
    DrawProjectedSphere(position,
                        tracker.r_ball,
                        tracker.rvec,
                        tracker.tvec,
                        tracker.calib,
                        frame,
                        color,
                        -1)



def main():
  camera = 'pixel2'  
  #camera = 'raspi'
  imageHeight = 360
  liveFeed = False

  dataDir = '/Users/kwatra/Home/pvt/robotx/RobotX2020VisionSystem/data'
  calibDir = os.path.join(dataDir, 'calib_data')
  inputDir = os.path.join(dataDir, 'target_videos')
  outputDir = os.path.join(dataDir, 'output')

  if liveFeed:
    videoSource = 0
  elif camera == 'pixel2':
    videoSource = os.path.join(inputDir, 'vision-tape-target-4.mp4')
  elif camera == 'raspi':
    #videoSource = os.path.join(inputDir, 'vision-tape-raspi-1.mov')
    videoSource = os.path.join(inputDir, 'vision-tape-raspi-2.mov')
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

  outputFile = os.path.join(outputDir,
      os.path.basename(videoSource) + '-' + calib.Id() + '-tracked.mp4')
  camera = cb.CameraSource(videoSource, calib.imageHeight, outputFile)

  tracker = ObjectTracker(calib)

  nFrames = 0
  while True:
    frame = camera.GetFrame()
    if frame is None:
      break
    
    gray, edges = tracker.Track(frame)
 
    if False:
      gray = np.stack((gray, gray, gray), axis=2)
      edges = np.stack((edges, edges, edges), axis=2)
      concat = np.concatenate((frame, gray), axis=0)

    ret, key = camera.OutputFrameAndTestContinue('Output', frame)

    #ret = True
    #ret = ret and camera.OutputFrameAndTestContinue('Output', frame)
    #ret = ret and camera.OutputFrameAndTestContinue('Output', concat)
    #ret = ret and cb.ShowFrameAndTestContinue('Gray frame', gray)
    #ret = ret and cb.ShowFrameAndTestContinue('Frame', frame)
    #ret = ret and cb.ShowFrameAndTestContinue('Edges', edges)
    if not ret:
      break;

    # Run simulation with different orientations depending on key pressed.
    simIndex = -1
    if key == 99:   # 'c' (current)
      simIndex = 0
    if key == 102:  # 'f' (front)
      simIndex = 1
    if key == 98:   # 'b' (back)
      simIndex = 2

    if simIndex >= 0 and tracker.trackingSuccess:
      ball = Ball()
      ball.StartSimulation(310.0, tracker, simIndex)
      while ball.t >= 0:
        sim_frame = frame.copy()
        ball.Simulate(tracker, 0.1)
        ball.Draw(tracker, sim_frame)
        # ShowFrame doesn't output to file.
        #if not cb.ShowFrameAndTestContinue('Output', sim_frame)[0]:
        if not camera.OutputFrameAndTestContinue('Output', sim_frame)[0]:
          break
    
    nFrames += 1



if __name__ == '__main__':
  main()
