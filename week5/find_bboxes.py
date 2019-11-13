# IMPORTS #
import cv2 as cv
import numpy as np
import pickle

from compute_mask import compute_mask

"""
This function computes the painting bounding box and its rotation.
Input parameters: background removal mask
"""

def compute_bbox_angle(contours, image):

    cor_1 = []
    cor_2 = []
    cor_3 = []
    cor_4 = []

    corners = []
    bbox_angles = []
    final_list = []

    """
    cv.namedWindow("mask", cv.WINDOW_NORMAL)
    cv.imshow("mask", im_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """

    # Loop over contours and find bounding boxes and their rotations
    for contour in contours:

        # Compute the minimum-area rotated bbox rectangle for a specific contour
        rect = cv.minAreaRect(contour)
        bbox = cv.boxPoints(rect)
        bbox = np.int0(bbox)

        cor_1.extend(bbox[3])
        cor_2.extend(bbox[2])
        cor_3.extend(bbox[1])
        cor_4.extend(bbox[0])

        corners.append(cor_4)
        corners.append(cor_3)
        corners.append(cor_2)
        corners.append(cor_1)

        # Rotation angle of detected bounding box
        alpha = rect[2]

        # Compute correct rotation angle
        if alpha < -45.0:
            alpha = (alpha*-1.0) + 90
        else:
            alpha = alpha*-1.0
        
        bbox_angles.append(alpha)
        bbox_angles.append(corners)
        final_list.append(bbox_angles)

        #print("Rotation angle of detected bbox: " + str(alpha))

        # Draw and show detected contours
        """
        cv.drawContours(image, [bbox] , -1, (0,255,0), 3)
        cv.namedWindow("contours", cv.WINDOW_NORMAL)
        cv.imshow("contours", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        """

        cor_1 = []
        cor_2 = []
        cor_3 = []
        cor_4 = []
        corners = []
        bbox_angles = []
    
    return final_list


"""
im = cv.imread('../qs/qsd1_w5/00005.jpg', cv.IMREAD_COLOR)
img = im
im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
_,_, contours = compute_mask(im, "hola", 'qsd1_w5')
final_list = compute_bbox_angle(contours, im)
print(final_list)
"""
"""
####### EXECUTION EXAMPLE ##################

# Frames ground truth
print("GROUND TRUTH")
frames = pickle.load(open('../qs/frames.pkl','rb'))
print(frames[2])

print("ACTUAL RESULTS")
# Read background removal mask
im = cv.imread('../week5/masks/00002.png', cv.IMREAD_COLOR)
# Ground truth (for testing the algorithm)
#im = cv.imread('../qs/qsd1_w5/00029.png', cv.IMREAD_COLOR)
final_list = compute_bbox_angle(im)
print(final_list)
"""