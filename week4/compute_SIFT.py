import cv2 as cv
import numpy as np

def compute_SIFT(img, merged_mask, size):

    # Create SIFT object
    sift = cv.xfeatures2d.SIFT_create()

    # Resize to speed up execution
    img = cv.resize(img, (size,size))

    if merged_mask is not None:
        merged_mask = cv.resize(merged_mask, (size,size))
        kp, des = sift.detectAndCompute(img, merged_mask)
    else:
        kp, des = sift.detectAndCompute(img, None)

    return des