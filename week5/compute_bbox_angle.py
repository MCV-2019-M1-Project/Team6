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
print(frames[1])

"""
Open masks from left to right. Save them first so they can be converted to grayscale.
Once you open the mask you can detect it with no problems.
"""
# Try with the mask
im = cv.imread('../qs/qsd1_w5/00008.png', cv.IMREAD_COLOR)
im = cv.medianBlur(im, 3)
im_gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
edges = cv.Canny(im, 50, 400, apertureSize=3)

############# FIND CONTOURS ############
height, width = im.shape[:2]
min_area = 0

print("Min area: " + str(min_area))

_, contours, hier = cv.findContours(im_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Orders contours by descending area 
sorted_contours = np.flip(sorted(contours, key=cv.contourArea))
# Keep only 10 biggest contours
# sorted_contours = sorted_contours[:20]
print("Number of detected contours: " + str(len(sorted_contours)))
# Store bounding boxes' areas
bbox_areas = np.zeros(len(sorted_contours))

for i in range(0, len(sorted_contours)):

    contour = sorted_contours[i]
    # Compute the minimum-area rotated bbox rectangle for a specific contour
    rect = cv.minAreaRect(contour)
    # Width and height of detected bounding box
    width = int(rect[1][1])
    height = int(rect[1][0])
    area = width*height

    if area < min_area:
        area = 0
    
    bbox_areas[i] = area
    # Transform detected bbox into points
    bbox = cv.boxPoints(rect)
    bbox = np.int0(bbox)
    # Bounding box corners
    px1, py1 = bbox[2]
    px2, py2 = bbox[3]
    px3, py3 = bbox[0]
    px4, py4 = bbox[1]

#print(bbox_areas)
#print(np.nonzero(bbox_areas))
#print(np.flip(sorted(bbox_areas)))


for i in range(0, len(bbox_areas)):

    if bbox_areas[i] != 0:
        contour_area = sorted_contours[i]
        print("Area: " + str(cv.contourArea(contour_area)))
        print("Area bbox: " + str(bbox_areas[i]))

        # Min area also delivers rotation angle of the bounding box detected in its third position!

        # COMPUTE BOUNDING BOXES PER EACH CONTOUR AREA, SORT THEM IN ORDER OF SIZE
        # PLOT BIGGEST BOUNDING BOXES
        # COMPUTE HEIGHT AND WIDTH OF BBOX TO KNOW AAREA
        rect = cv.minAreaRect(contour_area)
        bbox = cv.boxPoints(rect)
        bbox = np.int0(bbox)

        width = int(rect[1][1])
        height = int(rect[1][0])

        print("RECT: ")
        print(rect)
        print("WIDTH: ")
        print(width)
        print("HEIGHT: ")
        print(height)


        px1, py1 = bbox[2]
        px2, py2 = bbox[3]
        px3, py3 = bbox[0]
        px4, py4 = bbox[1]

        print(px1)
        print(py1)
        print(px2)
        print(py2)
        print(px3)
        print(py3)
        print(px4)
        print(py4)
 
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