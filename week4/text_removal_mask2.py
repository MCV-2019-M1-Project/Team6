# IMPORTS #
import cv2 as cv
import glob
import os
import math
import numpy as np
from matplotlib import pyplot as plt

def find_text(img, background_mask, name, option):
    
    #Option find is used previous to read the text in the image, whereas option remove should be used when text needs to be completely removed from it
    if option == 'find':
        max_aspect_ratio = 0.7
        min_aspect_ratio = 1/15
        min_occupancy_ratio = 0.5
        min_compactness_ratio = 0.0022
        th=0.004
    if option == 'remove':
        max_aspect_ratio = 0.6
        min_aspect_ratio = 1/20
        min_occupancy_ratio = 0.4
        min_compactness_ratio = 0.002
        th=0.005

    #Image pre-processing: the difference betwen the image opening and closing is computed
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*6-1, 2*6-1))
    img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    img_enhanced = img_closing - img_opening
    img_lowpass	= cv.GaussianBlur(img_enhanced, (5,5), 20)

    #Binarization: only the 4% brightest pixels are set to positive
    height = img.shape[0]
    width = img.shape[1]
    npx =  height*width
    hist = cv.calcHist([img_lowpass], [0], None, [256], [0, 255])/npx
    prob=0
    i=0
    while(prob<th):
        prob += hist[255-i][0]
        i+=1
    th=255-i
    ret,img_binary = cv.threshold(img_lowpass,th,255,cv.THRESH_BINARY)

    #Candidate text region extraction
    kernel = np.ones((15,30), np.uint8)
    img_dilated = cv.dilate(img_binary,kernel,iterations = 3)
    kernel = np.ones((10,10), np.uint8)
    img_eroded = cv.erode(img_dilated,kernel,iterations = 2)
    img_opening = cv.morphologyEx(img_eroded, cv.MORPH_OPEN, np.ones((15,15), np.uint8))

    num, labels = cv.connectedComponents(img_opening)
    sorted_labels = labels.ravel()
    sorted_labels = np.sort(sorted_labels)

    labels = np.uint8(labels)
    """
    cv.imshow("img_binary", labels*255)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    #Discarting non desired regions
    if np.shape(background_mask)[0]>1:
        min_size = npx/(20*20)
    else:
        min_size = npx/(15*15)
    coords = []

    for i in range(1, num+1):

        region = labels == i
        region = np.uint8(region)
        positions =  np.where(labels == i)
        size = len(positions[0])

        if size < min_size:

            labels[positions] = 0

        else:

            _, contours,_ = cv.findContours(region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
            largest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            (x,y,w,h) = cv.boundingRect(largest_contour)
            perimeter = cv.arcLength(largest_contour,True)
            if h/w>max_aspect_ratio or h/w<min_aspect_ratio or len(positions[0])/(w*h)<min_occupancy_ratio or (w*h)/(perimeter*perimeter)<min_compactness_ratio:
                labels[positions] = 0
            else:
                coords.append((x,y,w,h,size))
    
    # cv.namedWindow('img_binary',cv.WINDOW_NORMAL)
    # cv.resizeWindow('img_binary', 600,600)
    # cv.imshow("img_binary", labels*255)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    #Creating the expected number of masks by keeping the biggest bounding boxes
    coords.sort(key=lambda x:x[4], reverse=True)
    masks = []
    tx = []
    ty = []
    if option == 'find':
        length = np.shape(background_mask)[0]
    if option == 'remove':
        length = max(len(coords),np.shape(background_mask)[0])

    for i in range(length):
        if len(coords)>0:
            (x,y,w,h,size) = coords.pop(0)
            m = np.zeros((height,width),np.uint8)
            m[y:y + h, x:x + w] = 255
            masks.append(np.uint8(m))
            tx.append((x+w)/2)
            ty.append((y+h)/2)
        else:
            m = np.zeros((height,width),np.uint8)
            masks.append(np.uint8(m))

        # cv.namedWindow('mask',cv.WINDOW_NORMAL)
        # cv.resizeWindow('mask', 600,600)
        # cv.imshow("mask", masks[i])
        # cv.waitKey(0)
        # cv.destroyAllWindows()


    if option == 'remove':
        total_mask=np.zeros((height,width),np.uint8)
        total_masks = []
        for i in range(len(masks)):
            total_mask+=masks[i]
        for i in range(np.shape(background_mask)[0]):
            total_masks.append(total_mask)
        return total_masks

    
    #Matching text mask to background mask (only if the image contains 2 or more paintings)
    if np.shape(background_mask)[0]>1:
        cx = np.zeros(2)
        cy = np.zeros(2)
        for i in range(np.shape(background_mask)[0]):
            moments = cv.moments(background_mask[i])
            cx[i]=int(moments["m10"]/moments["m00"])
            cy[i]=int(moments["m01"]/moments["m00"])
        if np.sum(masks[0])!=0 and np.sum(masks[1])!=0:
            if math.sqrt(((cx[0]-tx[0])**2)+((cy[0]-ty[0])**2)) + math.sqrt(((cx[1]-tx[1])**2)+((cy[1]-ty[1])**2)) > math.sqrt(((cx[0]-tx[1])**2)+((cy[0]-ty[1])**2)) + math.sqrt(((cx[1]-tx[0])**2)+((cy[1]-ty[0])**2)):
                aux = masks[0]
                masks[0] = masks[1]
                masks[1] = aux
        else:
            if math.sqrt(((cx[0]-tx[0])**2)+((cy[0]-ty[0])**2)) > math.sqrt(((cx[1]-tx[0])**2)+((cy[1]-ty[0])**2)):
                aux = masks[0]
                masks[0] = masks[1]
                masks[1] = aux


    return masks

# for f in sorted(glob.glob(qs_l)):
#     name = os.path.splitext(os.path.split(f)[1])[0]
#     im = cv.imread(f, cv.IMREAD_COLOR)

"""
im = cv.imread('../qs/' + QUERY_SET + '/00005.jpg', cv.IMREAD_COLOR)
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
background_mask = [np.zeros((100,100))]
masks = find_text(im, background_mask)

cv.imshow("mask", masks[0])
cv.waitKey(0)
cv.destroyAllWindows()
""" 