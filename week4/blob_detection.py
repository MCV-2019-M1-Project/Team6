from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import cv2 as cv
import numpy as np

"""
im: gray image as input
keypoints: keypoint objects list

Execution example:
im = cv.imread('../qs/qsd1_w4/00016.jpg', cv.IMREAD_COLOR)
im_gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
"""

def laplacian_of_gaussian(im):

    blobs_log = blob_log(im, max_sigma=30, num_sigma=10, threshold=.1)

    # Compute radius of blobs
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2) 

    # Compute keyoints
    keypoints = []
    for i in range(0, len(blobs_log)):
        log_circle = blobs_log[i, :]
        keypoints.append(cv.KeyPoint(log_circle[0], log_circle[1], log_circle[2]))

    return keypoints

def difference_of_gaussian(im):

    blobs_dog = blob_dog(im, max_sigma=30, threshold=.1)
    # Compute radius of blobs
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2) 

    # Compute keyoints
    keypoints = []
    for i in range(0, len(blobs_dog)):
        log_circle = blobs_dog[i, :]
        keypoints.append(cv.KeyPoint(log_circle[0], log_circle[1], log_circle[2]))

    return keypoints

def determinant_of_hessian(im):

    blobs_doh = blob_doh(im, max_sigma=30, threshold=.01)
    # Compute radius of blobs
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2) 

    # Compute keyoints
    keypoints = []
    for i in range(0, len(blobs_doh)):
        log_circle = blobs_doh[i, :]
        keypoints.append(cv.KeyPoint(log_circle[0], log_circle[1], log_circle[2]))

    return keypoints


