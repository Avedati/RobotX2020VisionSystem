import cv2
import numpy as np

cursor_pos = None
def UpdateCursor(event, x, y, flags, param):
    global cursor_pos
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_pos = (x, y)


def DrawText(image, text, pos, color, scale=0.5/300):
    scale = scale * image.shape[1]
    if pos[0] < 1:
        pos = tuple([int(image.shape[1] * p) for p in pos])
    image2 = image.copy()
    cv2.putText(image2, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, lineType=cv2.LINE_AA)
    return image2


def DisplayColorAt(image, hsv, cursor_pos):
    if cursor_pos is not None:
        x, y = cursor_pos
        if x >= 0 and y >= 0 and x < hsv.shape[1] and y < hsv.shape[0]:
            color = hsv[y, x, :]
            image = DrawText(image, str(color), (0.05, 0.05), (255, 0, 0))
    return image


def DetectYellow(hsv, rgb):
    # Threshold the HSV image, keep only the color pixels
    hsv_mask = cv2.inRange(hsv, (20, 40, 100), (30, 255, 255))
    # Blue : Green ratio.
    rgb_mask = (rgb[:,:,0] < 0.7 * rgb[:,:,1]) * 255
    mask = hsv_mask & rgb_mask
    return np.atleast_3d(mask)


image = cv2.imread('image5.png')
image = cv2.medianBlur(image, 5)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", UpdateCursor)

while True:
  image2 = DisplayColorAt(image, hsv, cursor_pos)  
  image2 = image2.astype(float) / 255
  mask = DetectYellow(hsv, image).astype(float) / 255
  image2 = ((image2 * mask + image2 * 0.3 * (1 - mask)) * 255).astype(np.uint8)

  cv2.imshow('image', image2)
  k = cv2.waitKey(1) & 0xFF
  if k == ord('q'):
    break
