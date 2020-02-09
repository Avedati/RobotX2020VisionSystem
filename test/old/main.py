import cv2
import math
import numpy as np
import imutils
import os

def processPolygon(approx):
	
	for i in range(len(approx)):
		for j in range(i, len(approx)):
			if dist(approx[i][0], approx[j][0]) < 50:
				
	return approx

def visionTapeTest(contour):
	lst_cnt = np.squeeze(contour)
	if len(list(lst_cnt.shape)) == 1:
		return False
	for i in range(1, len(lst_cnt)):
		if lst_cnt[0,0] >= lst_cnt[i,0]:
			return False
	return True

def main():
	cap = cv2.VideoCapture('video.mp4')
	if cap.isOpened():
		while 1:
			ret, frame = cap.read()
			if frame is None:
				continue
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			blurred = cv2.GaussianBlur(gray, (5, 5), 0)
			thresholded = cv2.Canny(blurred, 150, 200)
			contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			approximations = []
			for i in range(len(contours)):
				epsilon = 0.01 * cv2.arcLength(contours[i], True)
				approx = cv2.approxPolyDP(contours[i], epsilon, True)
				approx = processPolygon(approx)
				if len(approx) < 8 or len(approx) > 9:
					continue
				if cv2.contourArea(approx) < 100:
					continue
				if not visionTapeTest(approx):
					continue
				approximations += [approx]

			approximations.sort(key=cv2.contourArea, reverse=True)

			if len(approximations) > 0:
				for approx in approximations:
					cv2.drawContours(frame, [approx], 0, (0, 255, 0), 4)
					for i, pt in enumerate(approx):
						cv2.putText(frame, str(i + 1), (pt[0][0], pt[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
			cv2.imshow('test', frame)

			k = cv2.waitKey(5) & 0xFF
			if k == 27:
				break

		cv2.destroyAllWindows();

if __name__ == '__main__':
	main()
