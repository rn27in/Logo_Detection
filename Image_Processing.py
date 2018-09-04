import cv2
import numpy as np

def classifier_image(img, pred):
    image = cv2.imread('../Test_Set/' + img +'.png')
    ##########Convert to gray scale image#########################
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ############Apply a Gaussian blur##########################
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)

    #################Apply Canny Edge Detection#############################
    edged = cv2.Canny(blurred, 30, 150)

    ##########Find contours###################3
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 7) &  (area > 300) ):
            contour_list.append(contour)

    if pred == 1:
        return len(contour_list)
    elif pred == 0:
        return "Yes" if len(contour_list)> 0 else "No"
    else:
        return 'Invalid input'