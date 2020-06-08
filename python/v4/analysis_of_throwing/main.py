import cv2
import math
import numpy as np
import time

ball_radius = 11 # Pixels
start_frame = 128 # First frame that the ball has left Ishansh's hand. 

def main():
	cap = cv2.VideoCapture('res/0.mp4')
	Cd_measurements = []
	prevPosX = -1
	prevPosY = -1
	prevVelX = -1
	prevVelY = -1
	prevDt = -1
	last_frame_time = time.time()
	if cap.isOpened():
		cap.set(1, start_frame)
		while True:
			ret, frame = cap.read()
			if ret == False:
				break
			try:
				blurred = cv2.GaussianBlur(frame, (5, 5), 1)
			except:
				break
			
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
					cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 3)

			ball = max(circles, key=lambda c: c[1])
			delta_time = time.time() - last_frame_time
			if not prevPosX == -1 and not prevPosY == -1 and delta_time != 0:
				velocityX = (ball[0][0] - prevPosX) / delta_time
				velocityY = (ball[0][1] - prevPosY) / delta_time
				velocity = np.sqrt(velocityX**2 + velocityY**2)
				if not prevVelX == -1 and not prevVelY == -1:
					accelerationX = (velocityX - prevVelX) / delta_time
					accelerationY = (velocityY - prevVelY) / delta_time
					acceleration = np.sqrt(accelerationX**2 + accelerationY**2)
					if velocity >= 0:
						Cd_measurement = (2/3) * ball_radius * acceleration / (velocity**2)
						if not math.isinf(Cd_measurement) and not Cd_measurement >= 10:
							Cd_measurements.append(Cd_measurement)
				prevVelX = velocityX
				prevVelY = velocityY

			prevPosX = ball[0][0]
			prevPosY = ball[0][1]
			
			cv2.imshow('test', frame)
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break

	if len(Cd_measurements) > 0:
		Cd_average = sum(Cd_measurements) / len(Cd_measurements)
		print(Cd_average)

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
