import cv2
import numpy as np

def drawcontours(img):
    '''
    This function is to draw contours on original image
    '''
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 7) & (area > 300)):
            contour_list.append(contour)

    img_copy = image.copy()
    cv2.drawContours(img_copy, contour_list, -1, (255, 0, 0), 2)
    cv2.imwrite('../img.png', img_copy)

