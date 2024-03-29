import cv2 as cv
import numpy as np

def compute_SURF(img, mask, size):
    """
    # Create SIFT object
    surf = cv.xfeatures2d.SURF_create(hessianThreshold=400)
    
    # Resize to speed up execution
    img = cv.resize(img, (size,size))

    if merged_mask is not None:
        merged_mask = cv.resize(merged_mask, (size,size))
        kp, des = surf.detectAndCompute(img, merged_mask)
    else:
        kp, des = surf.detectAndCompute(img, None)

    return des
    """
    # Create SURF object
    surf = cv.xfeatures2d.SURF_create(hessianThreshold=400)

    # Mask preprocessing
    if mask is not None:
        indices = np.where(mask != [0])
        if(indices[0].size != 0 and indices[1].size !=0):
            img = img[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
            mask = mask[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
        mask = mask.astype('uint8')
        img = cv.bitwise_and(img, img, mask=mask)

        img = cv.resize(img, (size,size))
        img = img[25:(size-25), 25:(size-25)]

    else:
        # Resize to speed up execution
        img = cv.resize(img, (size,size))
        img = img[25:(size-25), 25:(size-25)]

    # Detect and compute the features
    _, des = surf.detectAndCompute(img, None)


    return des