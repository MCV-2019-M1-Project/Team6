# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
from numpy import r_
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

def compute_hog(im, block_size):
    image = cv.resize(im, (128,128))

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(block_size, block_size),
                        cells_per_block=(5, 5), visualize=True, multichannel=True, feature_vector=True)

    return np.expand_dims(fd, axis=1)
