# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
from numpy import r_
import matplotlib.pyplot as plt
import pickle

"""
Detect bounding box containing paintings, its rotation angle and compute bounding box of detected and rotated painting (TO DO)
"""

"""
angles = pickle.load(open('../qs/angles_qsd1w5.pkl','rb'))
print(angles[26])
"""
frames = pickle.load(open('../qs/frames.pkl','rb'))
print(frames[19])

im = cv.imread('../qs/qsd1_w5/00019.jpg', cv.IMREAD_COLOR)
im = cv.medianBlur(im, 3)
im_gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
edges = cv.Canny(im, 50, 400, apertureSize=3)

############# FIND CONTOURS ############

areas_contours = []

_, contours, hier = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#print(hier)

# Get number of outer contours
# print(hier)

for i, c in enumerate(contours):
    area = cv.contourArea(c)
    areas_contours.append(area)

contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
#print(type(contour_sizes))

largest_contour = max(contour_sizes, key=lambda x: x[0])[1]
#print(largest_contour)
#print(type(largest_contour))
#print(largest_contour)

# Sort contours by area
sorted_area = sorted(zip(areas_contours, contours), key=lambda x: x[0], reverse=True)
sorted_contours = sorted(contours, key=cv.contourArea)

#print(sorted_contours[0])

contour_area = sorted_contours[len(sorted_contours)-1]
print("Area: " + str(cv.contourArea(contour_area)))

# Min area also delivers rotation angle of the bounding box detected in its third position!

# COMPUTE BOUNDING BOXES PER EACH CONTOUR AREA, SORT THEM IN ORDER OF SIZE
# PLOT BIGGEST BOUNDING BOXES
# COMPUTE HEIGHT AND WIDTH OF BBOX TO KNOW AAREA
rect = cv.minAreaRect(contour_area)
bbox = cv.boxPoints(rect)
bbox = np.int0(bbox)

print(bbox)
print(bbox[0][0] - bbox[1][0])

# Rotation angle of detected bounding box
rot_angle = rect[2]

#print(rot_angle)
if rot_angle < -45.0:
    rot_angle = (rot_angle*-1.0) + 90
else:
    rot_angle = rot_angle*-1.0

print("Rotation angle of detected bbox: " + str(rot_angle))

cv.drawContours(im, [bbox] , -1, (0,255,0), 3)

cv.namedWindow("contours", cv.WINDOW_NORMAL)
cv.imshow("contours", im)
cv.waitKey(0)
cv.destroyAllWindows()

######### HOUGH TRANSFORM #####################
"""
edges = cv.Canny(im, 50, 400, apertureSize=3)

cv.namedWindow("Edges Image", cv.WINDOW_NORMAL)
cv.imshow("Edges Image", edges)
cv.waitKey(0)
cv.destroyAllWindows()

lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=150, maxLineGap=80)
#lines = cv.HoughLines(edges, 1, np.pi/180, 150, None, 0, 0)

mean = np.mean(lines)
print(mean)
# Draw lines on the image
for line in lines:
    #print(line[0])
    x1, y1, x2, y2 = line[0]
    #print(x2-x1)
    cv.line(im, (x1, y1), (x2, y2), (255, 0, 0), 3)

# Show result
cv.namedWindow("Result Image", cv.WINDOW_NORMAL)
cv.imshow("Result Image", im)
cv.waitKey(0)
cv.destroyAllWindows()
"""