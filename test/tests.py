import cv2
import math
import numpy as np

def dot(x1, y1, x2, y2, x3, y3):
  v12 = (x2 - x1, y2 - y1)
  v23 = (x3 - x2, y3 - y2)
  return v12[0] * v23[0] + v12[1] * v23[1]

def dist(x1, y1, x2, y2):
  dx = x2 - x1
  dy = y2 - y1
  return math.sqrt(dx * dx + dy * dy)


"""def visionTapeTest(visionTape):
  points = [np.squeeze(x) for x in list(visionTape)]
  return np.asarray(points)
"""

def halfVisionTapeTest(visionTape):
  print('halfVision')
  points = list(visionTape)
  minIndex = 0
  for i in range(1, len(points)):
    if points[i][0] < points[minIndex][0]:
      minIndex = i

  print(len(points))
  
  for i in range(1, 4):
    if points[(minIndex+i)%len(points)][0] < points[(minIndex)%len(points)][0]:
      return None
  for i in range(0, 3):
    if points[(minIndex+i)%len(points)][0] > points[(minIndex+3)%len(points)][0]:
      return None
  for i in range(0, 4):
    if i == 1 or i == 2:
      continue
    if points[(minIndex+i)%len(points)][1] > points[(minIndex+1)%len(points)][1] and \
       points[(minIndex+i)%len(points)][1] > points[(minIndex+2)%len(points)][1]:
      return None

  return np.asarray(points)

def visionTapeTest(visionTape):
  points = list(visionTape)
  if len(points) != 8:
    return None
  minIndex = 0
  for i in range(1, len(points)):
    if points[i][0] < points[minIndex][0]:
      minIndex = i
  
  if points[minIndex][0] > points[(minIndex+1)%len(points)][0]:
    return None
  if points[(minIndex+1)%len(points)][0] > points[(minIndex+2)%len(points)][0]:
    return None
  if points[(minIndex+2)%len(points)][0] > points[(minIndex+3)%len(points)][0]:
    return None
  """
  for i in range(1, 4):
    if points[(minIndex+i)%len(points)][0] < points[(minIndex)%len(points)][0]:
      return None
  for i in range(0, 3):
    if points[(minIndex+i)%len(points)][0] > points[(minIndex+3)%len(points)][0]:
      return None
  """
  for i in range(0, 4):
    if i == 1 or i == 2:
      continue
    if points[(minIndex+i)%len(points)][1] > points[(minIndex+1)%len(points)][1] and \
       points[(minIndex+i)%len(points)][1] > points[(minIndex+2)%len(points)][1]:
      return None

  if points[(minIndex+7)%len(points)][0] > points[(minIndex+6)%len(points)][0]:
    return None
  if points[(minIndex+6)%len(points)][0] > points[(minIndex+5)%len(points)][0]:
    return None
  if points[(minIndex+5)%len(points)][0] > points[(minIndex+4)%len(points)][0]:
    return None
  """
  for i in range(4, 7):
    if points[(minIndex+i)%len(points)][0] < points[(minIndex+7)%len(points)][0]:
      return None
  for i in range(5, 8):
    if points[(minIndex+i)%len(points)][0] > points[(minIndex+4)%len(points)][0]:
      return None
  """
  for i in range(4, 8):
    if i == 5 or i == 6:
      continue
    if points[(minIndex+i)%len(points)][1] > points[(minIndex+5)%len(points)][1] and \
       points[(minIndex+i)%len(points)][1] > points[(minIndex+6)%len(points)][1]:
      return None

  new_points = [points[(minIndex+i)%len(points)] for i in range(len(points))]
  return np.asarray(new_points)


def processPolygon(polygon, num_corners):
  points = [np.squeeze(x) for x in list(polygon)]
  if len(points) < num_corners:
    return None
  cosines = []
  for i in range(1, len(points)+1):
    x1, y1 = polygon[i-1][0]
    x2, y2 = polygon[i%len(points)][0]
    x3, y3 = polygon[(i+1)%len(points)][0]
    l12 = dist(x1, y1, x2, y2)
    l23 = dist(x2, y2, x3, y3)
    if l12 * l23 > 0:
      cosine = dot(x1, y1, x2, y2, x3, y3) / (l12 * l23)
      cosines.append(cosine)

  indices = list(range(len(polygon)))
  indices.sort(key=lambda i: cosines[i])
  kept_indices = [indices[i] for i in range(num_corners)]
  kept_indices.sort()
  kept_points = [points[i] for i in kept_indices]
  
  return np.array(kept_points)





def processVisionTape(polygon):
  return processPolygon(polygon, 8)

  points = list(polygon)
  pointsToRemove = []
  for i in range(1, len(points)+1):
    x1, y1 = polygon[i-1][0]
    x2, y2 = polygon[i%len(points)][0]
    x3, y3 = polygon[(i+1)%len(points)][0]
    l12 = dist(x1, y1, x2, y2)
    l23 = dist(x2, y2, x3, y3)
    if l12 * l23 == 0:
      pointsToRemove += [i%len(points)]
    elif dot(x1, y1, x2, y2, x3, y3) / (l12 * l23) >= 0.9:
      pointsToRemove += [i%len(points)]
  points = [points[i] for i in range(len(points)) if i not in pointsToRemove]
  """pointsToRemove = []
  for i in range(len(points)):
    for j in range(i+1, len(points)):
      if dist(*points[i][0], *points[j][0]) <= 10:
        pointsToRemove += [j]
  points = [points[i] for i in range(len(points)) if i not in pointsToRemove]"""
  return np.array(points)

def polygonTests(polygon):
  return cv2.isContourConvex(polygon)
