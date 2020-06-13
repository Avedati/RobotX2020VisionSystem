import math
import numpy as np

def main():
	with open('/Users/spiderfencer/RobotX2020VisionSystem/data/output/ball-sim-pixel2-sphere4.mp4-720-25-ball-tracks.txt') as fp:
		lines = [line.strip() for line in fp.readlines()]
		frames = []
		for line in lines:
			parts = [float(part) for part in line.split(' ')]
			frames.append((int(parts[0]), (parts[1], parts[2], parts[3])))
			delta_x = frames[-1][1][0] - frames[0][1][0]
			delta_z = frames[-1][1][2] - frames[0][1][2]
			theta = math.atan2(delta_z, delta_x) - np.pi/2
		

if __name__ == '__main__':
	main()
