# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
from numpy import r_
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

"""
im = cv.imread('../qs/qsd1_w2/00003.jpg', cv.IMREAD_COLOR)
img_gray =  cv.cvtColor(im, cv.COLOR_BGR2GRAY)

cv.imshow("img", img_gray)
cv.waitKey(0)
cv.destroyAllWindows()

# Resize image to speed up execution
# resized_im = cv.resize(img_gray, (512,512))
"""

"""
_, hog_im = hog(resized_im, orientations = 8, pixels_per_cell = (16,16),
            cells_per_block=(1,1),visualize = True, multichannel=False)
"""

def compute_hog(im, block_size):

    hogs = []
    im = cv.resize(im, (256,256))
    # Variable to store histograms
    hist = []
    h, w = im.shape[:2]
    # Compute LBP dividing the image in blocks
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Block
            block = im[i:(i+block_size),j:(j+block_size)]
            _, hog_im = hog(block, orientations = 8, pixels_per_cell = (8,8),
                cells_per_block=(1,1),visualize = True, feature_vector=True, multichannel=False)
            hogs.append(np.hstack(hog_im))
            #print(np.hstack(hog_im))

    return np.hstack(hogs)

"""
im = cv.imread('../qs/qsd1_w2/00003.jpg', cv.IMREAD_COLOR)
img_gray =  cv.cvtColor(im, cv.COLOR_BGR2GRAY)
hogs = compute_hog(img_gray, 8)

print(np.hstack(hogs))
"""
"""
def extractDigits(lst): 
    res = [] 
    for el in lst: 
        #sub = el.split(', ') 
        res.append(el) 
      
    return(res) 

hola =[]
im = cv.imread('../qs/qsd1_w2/00003.jpg', cv.IMREAD_COLOR)
img_gray =  cv.cvtColor(im, cv.COLOR_BGR2GRAY)
hogs = compute_hog(img_gray, 8)

print(np.hstack(hogs))
print(len(np.hstack(hogs)))
"""

"""
print(hogs[0])
print(hogs[0][1].tolist())

one = hogs[0][1].tolist()
hola.append(one)
two = hogs[1][1].tolist()
hola.append(two)
"""

"""
final_list = []
for i in range(0, len(hogs) - 1):
    for j in range(0, len(hogs) - 1):
        elem = hogs[i][j].tolist()
        final_list.append(elem)
"""



"""
hogs = compute_hog(img_gray, 8)
print(type(hogs[0][1]))
print(hogs[0])

print(hogs)
"""


"""
print(hog_im)
print(np.nonzero(hog_im))
print(len(hog_im))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(im, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_im, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_im, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
"""
