import numpy as np
import cv2
import math

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

def processHexagon(polygon):
	polygon = np.squeeze(polygon)
	numPassed = 0
	pairs = []
	for i in range(len(polygon)):
		p1 = polygon[i]
		p2 = polygon[(i+1)%len(list(polygon))]
		p3 = polygon[(i+2)%len(list(polygon))]
		v12 = Vector2(p2[0] - p1[0], p2[1] - p1[1])
		v23 = Vector2(p3[0] - p2[0], p3[1] - p2[1])
		dotProduct = v12.dot(v23)
		lengths = v12.length() * v23.length()
		if lengths == 0:
			continue
		cosTheta = dotProduct / lengths
		pairs.append([cosTheta, (i+1)%len(polygon)])
	pairs.sort(key=lambda	k: k[0], reverse=True)
	result = [pairs[i][1] for i in range(6)]
	result2 = [list(polygon)[i] for i in result]
	distances = []
	for i in range(len(result2)):
		p1 = result2[i]
		p2 = result2[(i+1)%len(result2)]
		distances.append(dist(p1[0], p1[1], p2[0], p2[1]))
	distances.sort();
	return [distances[0], np.squeeze(result2)]

def processOctagon(polygon):
	polygon = np.squeeze(polygon)
	numPassed = 0
	pairs = []
	for i in range(len(polygon)):
		p1 = polygon[i]
		p2 = polygon[(i+1)%len(list(polygon))]
		p3 = polygon[(i+2)%len(list(polygon))]
		v12 = Vector2(p2[0] - p1[0], p2[1] - p1[1])
		v23 = Vector2(p3[0] - p2[0], p3[1] - p2[1])
		dotProduct = v12.dot(v23)
		lengths = v12.length() * v23.length()
		if lengths == 0:
			continue
		cosTheta = dotProduct / lengths
		pairs.append([cosTheta, (i+1)%len(polygon)])
	pairs.sort(key=lambda	k: k[0], reverse=True)
	if len(list(polygon)) < 8:
		return None
	result = [pairs[i][1] for i in range(8)]
	result2 = [list(polygon)[i] for i in result]
	distances = []
	for i in range(len(result2)):
		p1 = result2[i]
		p2 = result2[(i+1)%len(result2)]
		distances.append(dist(p1[0], p1[1], p2[0], p2[1]))
	distances.sort();
	return [distances[0], np.squeeze(result2)]

def polygonTests(polygon):
	polygon = np.squeeze(polygon)
	if len(polygon.shape) != 2:
		return -1
	totalDistance = dist(polygon[0,0], polygon[0,1], polygon[len(polygon)-1,0], polygon[len(polygon)-1,1])
	distances = []
	for i in range(len(list(polygon)) - 1):
		d = dist(polygon[i,0], polygon[i,1], polygon[i+1,0], polygon[i+1,1])
		distances.append(d)
		totalDistance += d
	averageDistance = totalDistance / len(list(polygon))
	flag = True
	for i in range(len(distances)):
		if abs(averageDistance - distances[i]) > averageDistance * 4 / 3:
			flag = False
			break

	if cv2.isContourConvex(polygon):
		return -1
	if len(polygon) < 6:
		return -1
	M = cv2.moments(polygon, False)
	if M["m00"] == 0:
		return -1
	y = int(M["m01"] / M["m00"])
	if y < HEIGHT / 2:
		return 6
	return 8

def main():
	camera = cv2.VideoCapture("/Users/spiderfencer/Desktop/opencv-test-video.mov")
	WIDTH = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
	HEIGHT = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
	nFrames = 0
	savedHexagons = []
	frames = []
	if camera.isOpened():
		while True:
			nFrames += 1
			ret, frame = camera.read()
			if frame is None:
				break
			frame = cv2.resize(frame, (WIDTH, HEIGHT))
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
				elif result == 6:
					hexagonPair = processHexagon(polygon)
					if not cv2.isContourConvex(hexagonPair[1]):
						continue
					hexagons.append(hexagonPair)
				elif result == 8:
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
			#	savedHexagons.append(hexagonsToDraw[0])
			#	frames.append(frame.copy())

			for i in range(min(1, len(hexagonsToDraw))):
				cv2.drawContours(frame, hexagonsToDraw, i, (0, 255, 0), 1)
				for j in range(len(hexagonsToDraw[i])):
					cv2.putText(frame, str(j+1), tuple(hexagonsToDraw[i][j]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
			for i in range(min(1, len(octagonsToDraw))):
				cv2.drawContours(frame, octagonsToDraw, i, (0, 0, 255), 1)
			cv2.imshow('test', frame)
			k = cv2.waitKey(5) & 0xFF
			if k == 27:
				break

			#if nFrames >= 1000:
			#	break

		objectsPoints = []
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
