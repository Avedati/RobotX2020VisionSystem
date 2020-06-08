"""
Instructions:
- Run the program with an argument from 0-11, which refers to the name of the video that you would like to examine from the res folder.
- Press the `p` key to play or pause the video.
- While paused:
  - Press the `i` key to input a frame number to go to in the video (the first frame is index 0, input is done through the command line).
  - Press the `x` key to get the x position of the ball in the frame.
  - Press the `y` key to get the y position of the ball in the frame.
  - Press the `g` key to get the current frame index.
  - Press the `v` key to approximate the instantaneous velocity of the ball at the current frame (approximation is made using the next frame).
"""

import cv2
import numpy as np
import sys

def getFrame(cap, index):
	orig_index = cap.get(1)
	cap.set(1, index)
	ret, frame = cap.read()
	cap.set(1, orig_index)
	return frame

def detectBall(frame):
	try:
		blurred = cv2.GaussianBlur(frame, (5, 5), 1)
	except:
		return (-1, -1), -1

	# https://stackoverflow.com/questions/20732158/red-and-yellow-triangles-detection-using-opencv-in-python
	# https://stackoverflow.com/questions/9179189/detect-yellow-color-in-opencv/19488733
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# https://stackoverflow.com/questions/9179189/detect-yellow-color-in-opencv/19488733
	mask_yellow = cv2.inRange(hsv,np.array((23, 130, 120)),np.array((27, 255, 255)))
	cutout = cv2.bitwise_and(frame, frame, mask=mask_yellow)
	kernel = np.ones((5, 5), np.uint8)
	dilation = cv2.dilate(cutout, kernel, iterations=2)
	erosion = cv2.erode(dilation, kernel, iterations=2)

	bgr = cv2.cvtColor(erosion, cv2.COLOR_HSV2BGR)
	gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
	contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	circles = []
	for contour in contours:
		approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
		if len(approx) >= 3:
			(x, y), radius = cv2.minEnclosingCircle(contour)
			circles.append(((x, y), radius))

	ball = max(circles, key=lambda c: c[1])
	return ball
	

def main():
	fp = 'res/0.mp4'
	if len(sys.argv) > 1:
		fp = 'res/{}.mp4'.format(sys.argv[1])
	cap = cv2.VideoCapture(fp)
	if cap.isOpened():
		while True:
			ret, frame = cap.read()
			if ret == False:
				break
			cv2.imshow('test', frame)
			k = cv2.waitKey(1) & 0xff
			if k == 27:
				break
			# https://stackoverflow.com/questions/38064777/use-waitkey-in-order-pause-and-play-video
			elif k == ord('p'):
				while True:
					k2 = cv2.waitKey(1) or 0xff
					cv2.imshow('test', frame)
					if k2 == ord('i'):
						frame_no = int(input('--> '))
						cap.set(1, frame_no)
						ret, frame = cap.read()
					elif k2 == ord('x'):
						(x, y), r = detectBall(frame)
						print(x)
					elif k2 == ord('y'):
						(x, y), r = detectBall(frame)
						print(y)
					elif k2 == ord('g'):
						frame_no = cap.get(1)
						print(frame_no)
					elif k2 == ord('v'):
						frame1 = getFrame(cap, cap.get(1) + 1)
						(x1, y1), r1 = detectBall(frame)
						(x2, y2), r2 = detectBall(frame1)
						if x1 == -1 or x2 == -1 or y1 == -1 or y2 == -1 or r1 == -1 or r2 == -1:
							continue
						fps = cap.get(cv2.CAP_PROP_FPS)
						if fps == 0:
							continue
						dt = 1/fps
						if dt != 0:
							print(np.sqrt((x1 - x2)**2 + (y1 - y2)**2) / dt)
					elif k2 == ord('p'):
						break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
