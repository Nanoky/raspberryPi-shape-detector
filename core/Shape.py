import cv2 as cv
import numpy as np
from core.Image import *

class Shape:
    """
    This class allow us to detect different shapes in a image
    """
    text_color = (10, 10, 255)
    contours_color = (10, 10, 255)

    epsilon = 0.05

    area_min = 3000
    area_max = 50000

    def __init__(self):
        self.image = Image()

    def getLabel(self, img, cnt):

        contours, _ = cv.findContours(cnt, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        labels = []
        
        for c in contours:
            area = cv.contourArea(c)

            if area > self.area_min and area < self.area_max:
                approx = cv.approxPolyDP(c, self.epsilon*cv.arcLength(c, True), True)
                cv.drawContours(img, [approx], 0, self.contours_color, 5)
                x = approx.ravel()[0]
                y = approx.ravel()[1] - 5
                label = ""
                if len(approx) == 3:
                    label = "triangle"
                elif len(approx) == 4:
                    x1, y1, w, h = cv.boundingRect(approx)
                    aspect_ratio = float(w) / float(h)
                    #print(aspect_ratio)
                    if aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
                        label = "Square"
                    else:
                        label = "rectangle"
                elif len(approx) == 10:
                    label = "star"
                else:
                    label = "Circle"
                
                cv.putText(img, label, (x, y),cv.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 1)

                labels.append(label)

        cv.imshow("Gray", cv.resize(img, (200, 200)))

        return labels


