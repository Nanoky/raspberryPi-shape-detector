import cv2 as cv
import numpy as np

class Image:

    def getHsvEqualized(self, img):
        img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

        img_yuv_eq = img_yuv
        #img_yuv_eq[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv_eq[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_yuv_eq = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

        img_hsv = cv.cvtColor(img_yuv_eq, cv.COLOR_BGR2HSV)

        return img_hsv

    def getGray(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def getHSVColorRange(self, img, hsv_min, hsv_max):
        return cv.inRange(img, hsv_min, hsv_max)

    def getImgOnlyContours(self, img, blur_rate):
        img_medBlur = cv.medianBlur(img, blur_rate)

        gauss = cv.adaptiveThreshold(img_medBlur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

        kernel = np.ones((5, 5))
        image = cv.dilate(gauss, kernel, iterations=1)

        return gauss

    def getImgOnlyContoursCanny(self, img, blur_rate, th1, th2):
        img_medBlur = cv.medianBlur(img, blur_rate)

        canny = cv.Canny(img_medBlur, th1, th2)

        kernel = np.ones((5, 5))
        image = cv.dilate(canny, kernel, iterations=1)

        return image