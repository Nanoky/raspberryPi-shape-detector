import cv2, numpy as np, os
from core.Shape import *


# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

# Create a window named trackbars.
cv2.namedWindow("Trackbars")


# Now create 6 trackbars that will control the lower and upper range of 
# H,S and V channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("Threshold1", "Trackbars", 60, 500, nothing)
cv2.createTrackbar("Threshold2", "Trackbars", 100, 500, nothing)
cv2.createTrackbar("MedianBlur", "Trackbars", 4, 50, nothing)

shape = Shape()
image = Image()

cap = cv2.VideoCapture(0)

img_size = 1000

while (True):
    #read image from video, create a copy to draw on
    _, img= cap.read()
    imgc = img.copy()

    #Get the new values of the trackbar in real time as the user changes 
    # them
    th1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    th2 = cv2.getTrackbarPos("Threshold2", "Trackbars")

    m = cv2.getTrackbarPos("MedianBlur", "Trackbars")
    if m % 2 == 0:
        m = m + 1
        
    imgcnt = image.getGray(imgc)
    imgcnt = image.getImgOnlyContoursCanny(imgcnt, m, th1, th2)
    cv.imshow("Gray_Processed", cv.resize(imgcnt, (200, 200)))
    
    labels = shape.getLabel(imgc, imgcnt)

    cv2.imshow("Image", cv2.resize(imgc, (200, 200))) #expect 2 frames per second

    k = cv2.waitKey(100)
    if k == ord('m'): 
        break
    if k == ord('s'):
        print(labels)


cap.release()
cv2.destroyAllWindows()