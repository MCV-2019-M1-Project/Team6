"""
TEXT REMOVAL USING MORPHOLOGY
"""

# IMPORTS
import glob
import cv2 as cv
import numpy as np
import os
import pandas as pd


QUERY_SET='qsd1_w2'
images = glob.glob('./' + QUERY_SET + '/*.jpg')
print(str(QUERY_SET) + " has " + str(len(images)) + " images")

# Structuring element
strel = np.ones((15,15), np.uint8)

# Structuring element used after text removal
strel_pd = np.ones((20,20),np.uint8)

# Number of columns considered from the center of the image towards the right
num_cols = 6

for f in sorted(images):
    # Read image
    name = os.path.splitext(os.path.split(f)[1])[0]
    img = cv.imread(f, cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Obtain image dimensions
    height,width = img_gray.shape[:2]
    
    # Boundaries of the analyzed area
    min_a = round(width/2)
    max_a = round(min_a + num_cols)
    
    # Store pixel values in the analyzed area
    counts_t = np.zeros(max_a-min_a)
    values_t = np.zeros(max_a-min_a)
    
    i = 0
    
    for p in range(min_a, max_a):
        # Per each column, compute number of ocurrences of every pixel value
        col = img_gray[:,p]
        
        # Pixel values and number of ocurrences for the whole column
        values = pd.Series(col).value_counts().keys().tolist()
        # counts = pd.Series(col).value_counts().tolist()
        
        values_t[i] = values[0]
        # counts_t[i] = counts[0]
       
        i += 1
    
    level = np.mean(values_t)
    
    if level < 128:
        final_img = cv.morphologyEx(img_gray, cv.MORPH_OPEN, strel)
        cv.imwrite('results/' + name + '_opening.png', final_img)
    elif level > 128:
        final_img = cv.morphologyEx(img_gray, cv.MORPH_CLOSE, strel)
        cv.imwrite('results/' + name + '_closing.png', final_img)
    
    # Create mask to identify bounding box area
    if level < 128:
        mask = (final_img != level)
        mask = mask.astype(np.uint8)
        mask *= 255
        mask = cv.bitwise_not(mask)
    elif level > 128:
        mask = (final_img == level)
        mask = mask.astype(np.uint8)
        mask *= 255
    
    cv.imwrite('results/' + name + '_mask.png', mask)

# =============================================================================
#     print(str(name))
#     print(values_t)
#     print(counts_t)
# =============================================================================


