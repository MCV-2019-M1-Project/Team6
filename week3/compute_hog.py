# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
from numpy import r_
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

def compute_hog(im, mask, block_size, img_size):

    # Mask preprocessing
    if mask is not None:
        indices = np.where(mask != [0])
        if(indices[0].size != 0 and indices[1].size !=0):
            im = im[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
            mask = mask[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
        mask = mask.astype('uint8')
        im = cv.bitwise_and(im, im, mask=mask)

    # Resize image for computation improvement and dimensions
    image = cv.resize(im, (img_size,img_size))

    # Compute FOG feature
    fd = hog(image, orientations=8, pixels_per_cell=(block_size, block_size),
                        cells_per_block=(5, 5), visualize=False, multichannel=True, feature_vector=True)

    return np.expand_dims(fd, axis=1).tolist()
