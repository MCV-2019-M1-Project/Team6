# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
from numpy import r_
import matplotlib.pyplot as plt


im = cv.imread('../qs/qsd1_w2/00003.jpg', cv.IMREAD_COLOR)
img_gray =  cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# Extract image dimensions
h, w = img_gray.shape[:2]

# DCT block size
block_size = 8

# Number of DCT coefficients to keep
num_coeffs = 8

# Resize image in case it is odd
if (h % 2 == 1):
    h += 1

if (w % 2 == 1):
    w += 1

resized_img = cv.resize(img_gray, (w, h)) 

# Remove salt and pepper noise
# filtered_im = cv.medianBlur(resized_img, 3)

filtered_im = resized_img

cv.imshow("image", filtered_im)
cv.waitKey(0)
cv.destroyAllWindows()

# Normalize filtered image
im_float = np.float32(filtered_im)/255.0
# Compute dct coefficients
im_dct = cv.dct(im_float)
# Convert to original format
img = np.uint8(im_dct)*255.0
# Variable to store DCT
dct_im = np.zeros([h,w], dtype=np.uint8)

print("Num of blocks" + str((h*w)/block_size))

for i in r_[:h:block_size]:
    for j in r_[:w:block_size]:
        # Compute DCT of current block
        block_float = np.float32(filtered_im[i:(i+8),j:(j+8)])
        block_dct = cv.dct(block_float)
        dct_im[i:(i+8),j:(j+8)] = np.uint8(block_dct)*255.0
        dct_block = dct_im[i:(i+8),j:(j+8)]
        #Zig-zag scan of DCT block
        dct_coeffs = np.concatenate([np.diagonal(dct_block[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dct_block.shape[0], dct_block.shape[0])])
        # Keep N coefficients
        dct_coeffs = dct_coeffs[0:num_coeffs]
        print(dct_coeffs)

cv.imshow("image", dct_im)
cv.waitKey(0)
cv.destroyAllWindows()

"""
# Display entire DCT
plt.imshow(dct_im,cmap='gray',vmax = np.max(dct_im)*0.01,vmin = 0)
plt.title( "8x8 DCTs of the image")
plt.show()

print(img)
"""