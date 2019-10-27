# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
from numpy import r_
import matplotlib.pyplot as plt

# descriptor = compute_dct(img_gray, 8, 16)

def compute_dct(im, block_size, num_coeffs):
    # List to store kept DCT coefficients
    dct_coeffs_list = []
    # Resize image to speed up execution
    im = cv.resize(im, (512, 512)) 
    h, w = im.shape[:2]
    # Variable to store DCT
    dct_im = np.zeros([h,w], dtype=np.uint8)

    #print("Num of blocks" + str((256*256)/(block_size*block_size)))

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Compute DCT of current block
            block = im[i:(i+block_size),j:(j+block_size)]
            block_float = np.float32(block)
            block_dct = cv.dct(block_float)
            dct_im[i:(i+block_size),j:(j+block_size)] = np.uint8(block_dct)*255.0
            dct_block = dct_im[i:(i+block_size),j:(j+block_size)]

            #Zig-zag scan of DCT block
            dct_coeffs = np.concatenate([np.diagonal(dct_block[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dct_block.shape[0], dct_block.shape[0])])

            # Keep N coefficients
            dct_coeffs = dct_coeffs[0:num_coeffs]
            dct_coeffs_list = np.append(dct_coeffs_list, dct_coeffs)

    return np.expand_dims(dct_coeffs_list, axis=1)