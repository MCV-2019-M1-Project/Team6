# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

"""
im: gray_scale image
block_size: size of blocks to divide the image
n_bins: number of bins to compute the histogram
n_points: number of neigbours for LBP 
radius: radious of LBP element
METHOD: 'uniform'
"""

def compute_lbp(im, block_size, n_bins, n_points, radius, METHOD):
    # Resize image to speed up execution
    im = cv.resize(im, (128,128))
    # Variable to store histograms
    hist = []
    h, w = im.shape[:2]
    # Compute LBP dividing the image in blocks
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Block
            block = im[i:(i+block_size),j:(j+block_size)]
            # Compute the LBP for the block
            block_lbp = np.float32(local_binary_pattern(block, n_points, 1))
            # Compute histogram over the block and normalize
            npx = block_lbp.shape[0]*block_lbp.shape[1]
            hist_lbp = cv.calcHist([block_lbp], [0], None, [n_bins], [0, 255])/npx
            hist.append(hist_lbp)

    return hist
