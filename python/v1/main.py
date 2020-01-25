import math
import numpy as np
import cv2
import time

def dist(p1, p2):
	return math.sqrt(math.pow(p1[0][0] - p2[0][0], 2) + math.pow(p1[0][1] - p2[0][1], 2))

def test(polygon, n):
	if cv2.contourArea(polygon) < 50:
		return False
	polygon = list(polygon)
	i = 0
	while i < len(polygon):
		if dist(polygon[i], polygon[(i+1)%len(polygon)]) < 10:
			if i + 1 == len(polygon):
				del polygon[i]
				break
			del polygon[i + 1]
		else:
			i += 1
	return len(polygon) == n

def contourCenter(cnt):
	M = cv2.moments(cnt)
	return [[0, 0] if M["m00"] == 0 else [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]]

def main():
	camera = cv2.VideoCapture("/Users/spiderfencer/Desktop/opencv-test-video.mov")
	# https://stackoverflow.com/questions/39953263/get-video-dimension-in-python-opencv/39953739
	width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
	height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
	prevHexagon = None
	prevOctagon = None
	previousHexagons = []
	previousOctagons = []
	startTime = time.time()
	if camera.isOpened():
		running = True
		while running:
			ret, frame = camera.read()
			if frame is None:
				break
			output = frame.copy()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			blurred = cv2.GaussianBlur(gray, (3, 3), 0)
			canny = cv2.Canny(blurred, 10, 160)
			# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
			kernel = np.ones((3, 3), np.uint8)
			dilated = cv2.dilate(canny, kernel)
			eroded = cv2.erode(dilated, kernel)
			contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
			for i in range(len(contours)):
				epsilon = 0.01*cv2.arcLength(contours[i], True)
				approx = cv2.approxPolyDP(contours[i], epsilon, True)
				#if len(previousHexagons) > 10 and len(previousOctagons) > 10:
				#	print('Time to calibrate: {} seconds'.format(time.time() - startTime))
				#	running = False
				#	break
				if (cv2.isContourConvex(approx) and (len(approx) == 6 or len(approx) == 8)):
					if len(approx) in [6,8] and test(approx, len(approx)):
						if len(approx) == 6:
							if prevHexagon is not None:
								if dist(contourCenter(approx), contourCenter(prevHexagon)) < 10:
									cv2.drawContours(output, [approx], 0, (0, 0, 255), 5)
							else:
								cv2.drawContours(output, [approx], 0, (0, 0, 255), 5)
							prevHexagon = approx
							previousHexagons.append(approx)
						if len(approx) == 8:
							if prevOctagon is not None:
								if dist(contourCenter(approx), contourCenter(prevOctagon)) < 10:
									cv2.drawContours(output, [approx], 0, (255, 0, 0), 5)
							else:
								cv2.drawContours(output, [approx], 0, (255, 0, 0), 5)
							prevOctagon = approx
							previousOctagons.append(approx)
			cv2.imshow('test', output)
			k = cv2.waitKey(5) & 0xFF
			if k == 27:
				break
		if len(previousHexagons) > 10 and len(previousOctagons) > 10:
			#_, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera("... calibration parameters ...")
			pass
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
