# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
from numpy import r_
import matplotlib.pyplot as plt
import pickle
import imutils

from compute_mask import compute_mask

"""
This function checks whether the paintings on an image are rotated or not and returns
the rotation angle.
"""

def check_rotation(image):

    # Compute edges on grayscale image
    edges = cv.Canny(image, 50, 400, apertureSize=3)
    # Compute contours on image
    _, contours,_ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours by descending area
    sorted_contours = np.flip(sorted(contours, key=cv.contourArea))
    # Get contour with biggest area to find the paintings' rotation
    contour = sorted_contours[0]

    # Compute the minimum-area rotated bbox rectangle for a specific contour
    rect = cv.minAreaRect(contour)

    # Variables to shaw found biggest contour
    bbox = cv.boxPoints(rect)
    bbox = np.int0(bbox)
    #cv.drawContours(im, [bbox] , -1, (0,255,0), 3)

    # Compute rotation angle of painting's
    alpha = rect[2]

    if 0.0 >= alpha < -45.0:
        alpha = (alpha*-1.0) + 90
    else:
        alpha = alpha*-1.0

    """
    cv.namedWindow("Result Image", cv.WINDOW_NORMAL)
    cv.imshow("Result Image", im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """

    return alpha

def rotate_image(im, alpha):
    if alpha > 90:
        alpha = 180+alpha

    rotated = imutils.rotate_bound(im, alpha)

    return rotated

def compute_coordinates(img_gray,alpha):
    rotated_gray = imutils.rotate_bound(img_gray, alpha)
    rotated_2_gray = imutils.rotate(rotated_gray, alpha)

    _,thresh = cv.threshold(rotated_2_gray,1,255,cv.THRESH_BINARY)
    _, contours,_= cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv.boundingRect(cnt)

    return x,y,w,h

"""
############ EXECUTION EXAMPLE #########################
im = cv.imread('../qs/qsd1_w5/00001.jpg', cv.IMREAD_COLOR)
im = cv.medianBlur(im, 3)
img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im = cv.cvtColor(im, cv.COLOR_BGR2HSV)

alpha = check_rotation(img_gray)
rotated = rotate_image(im, alpha)
cv.namedWindow("Rotated (Correct)", cv.WINDOW_NORMAL)
cv.imshow("Rotated (Correct)", rotated)
cv.waitKey(0)
cv.destroyAllWindows()
x,y,w,h = compute_coordinates(img_gray)

_,_, contours = compute_mask(rotated, im, "hola", 'qsd1_w5', alpha, x,y,w,h)
"""


"""
for f in sorted(glob.glob(qs_l)):
    #im = cv.imread('../qs/qsd1_w5/00010.jpg', cv.IMREAD_COLOR)
    im = cv.imread(f, cv.IMREAD_COLOR)
    height, width = im.shape[:2]
    print(height)
    print(width)
    #im = cv.imread(f, cv.IMREAD_COLOR)
    im = cv.medianBlur(im, 3)
    img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.cvtColor(im, cv.COLOR_BGR2HSV)

    cv.namedWindow("Result Image", cv.WINDOW_NORMAL)
    cv.imshow("Result Image", img_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()
    alpha = check_rotation(img_gray)
    if alpha > 90:
        alpha = 180+alpha

    print(alpha)
    rotated = imutils.rotate_bound(img_gray, alpha)
    rotated_2 = imutils.rotate(rotated, alpha)

    cv.namedWindow("Rotated (Correct)", cv.WINDOW_NORMAL)
    cv.imshow("Rotated (Correct)", rotated)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.namedWindow("Rotated (Correct)", cv.WINDOW_NORMAL)
    cv.imshow("Rotated (Correct)", rotated_2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    _,thresh = cv.threshold(rotated_2,1,255,cv.THRESH_BINARY)
    _, contours,_= cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv.boundingRect(cnt)
    crop = rotated_2[y:y+h,x:x+w]
    crop = cv.resize(crop, (width,height))
    height, width = crop.shape[:2]
    print(height)
    print(width)

    cv.namedWindow("Result Image", cv.WINDOW_NORMAL)
    cv.imshow("Result Image", crop)
    cv.waitKey(0)
    cv.destroyAllWindows()
"""