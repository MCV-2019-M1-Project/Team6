import cv2 as cv
import numpy as np

def compute_ORBS(img, merged_mask, size):

    # Initiate ORB detector
    orb = cv.ORB_create()

    # Resize to speed up execution
    img = cv.resize(img, (size,size))

    if merged_mask is not None:
        merged_mask = cv.resize(merged_mask, (size,size))
        kp, des = orb.detectAndCompute(img, merged_mask)
    else:
        kp, des = orb.detectAndCompute(img, merged_mask)

    return des