import cv2
import sys
import os

filepath = sys.argv[1]
cam = cv2.VideoCapture(filepath)

try:
    if not os.path.exists("data"):
        os.makedirs("data")
except OSError:
    print("Error: Creating directory of data")

currentframe = 0

while(True):

        ret,frame = cam.read()

        if ret:
            name = name = f"./data/frame{currentframe}.jpg"
            print(f"Creating... {name}")
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
cam.release()
cv2.destroyAllWindow()
