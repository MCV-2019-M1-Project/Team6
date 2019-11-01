import cv2 as cv
import numpy as np
from compute_mask import compute_mask
from text_removal_mask2 import find_text
from matplotlib import pyplot as plt
import glob
import os

def compute_SIFT_kp_and_des(img, bg_mask, text_mask, sift):

    if bg_mask is not None:
        prod = cv.bitwise_not(text_mask) * bg_mask
        prod = prod.astype(np.uint8)
        kp, des = sift.detectAndCompute(img, prod)
    else:
        kp, des = sift.detectAndCompute(img, None)

    return kp, des

sift = cv.xfeatures2d.SIFT_create()
descriptors_db = []
descriptors_q = []
keypoints_db = []

i=0

for f in sorted(glob.glob('../database/*.jpg')):
        # Read image
        img = cv.imread(f, cv.IMREAD_COLOR)
        # Resize to speed up execution
        img = cv.resize(img, (512,512))
        img = cv.medianBlur(img, 3)
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        kp, des = compute_SIFT_kp_and_des(img_gray, None, None, sift)
        descriptors_db.append(des)
        keypoints_db.append(kp)
        i+=1
        print(str(i))


QUERY_SET = 'qsd1_w4'
qs_l = '../qs/' + QUERY_SET + '/*.jpg'

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

for f in sorted(glob.glob(qs_l)):
        # Read image 
        name = os.path.splitext(os.path.split(f)[1])[0]
        im = cv.imread(f, cv.IMREAD_COLOR)
        # Resize to speed up execution
        im = cv.resize(im, (512,512))
        im = cv.medianBlur(im, 3)
        im_gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
        # Compute background and text masks
        bg_mask,_ = compute_mask(im, "prova" + name, 'qsd1w4')
        text_mask = find_text(im_gray, bg_mask, "provatext" + name)
        # Check whether the image contains two paintings
        length = np.shape(bg_mask)[0]

        if length > 2:
            length = 1
            bg_mask = [bg_mask]
            text_mask = [text_mask]

        for m in range(length):
            print("Query image: " + name)
            kp, des = compute_SIFT_kp_and_des(img_gray, bg_mask[m], text_mask[m], sift)
            descriptors_q.append(des)
            print(str(i))
            # Store matches
            matches_final = np.zeros(279)
            h = 0
            for f in sorted(glob.glob('../database/*.jpg')):

                # Read every Database image to plot keypoints matching
                im_db = cv.imread(f, cv.IMREAD_COLOR)
                im_db = cv.resize(im_db, (512,512))
                im_db = cv.medianBlur(im_db, 3)
                img_gray_db = cv.cvtColor(im_db,cv.COLOR_BGR2GRAY)

                matches = flann.knnMatch(descriptors_db[h], des, k=2)
                # Need to draw only good matches, so create a mask
                matchesMask = [[0,0] for l in range(len(matches))]

                # ratio test as per Lowe's paper
                matches_good = 0

                for k,(m,n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
                        matchesMask[k]=[1,0]
                        matches_good += 1

                # Number of "true" matches
                matches_final[h] = matches_good
                """
                # Matches drawing
                draw_params = dict(matchColor = (0,255,0),
                            singlePointColor = (255,0,0),
                            matchesMask = matchesMask,
                            flags = cv.DrawMatchesFlags_DEFAULT)

                # Plot matches
                img3 = cv.drawMatchesKnn(im_gray,kp,img_gray_db,keypoints_db[h],matches,None,**draw_params)
                plt.imshow(img3,)
                plt.show()
                """
                h+=1
            
            ranking = sorted(range(len(matches_final)), key=lambda i: matches_final[i], reverse=True)[:5]
            numbers =  sorted( [(x,i) for (i,x) in enumerate(matches_final)], reverse=True )[:5] 
            
            #num = str(np.argmax(matches_final))
            print("Most similar in DDBB: " + str(ranking) + "matches" + str(numbers))

#print(len(descriptors_q))

####################### OLD CODE ########################3333
"""
# Read image and its DDBB correspondence
im = cv.imread('../qs/qsd1_w4/00016.jpg', cv.IMREAD_COLOR)
im_db = cv.imread('../database/bbdd_00026.jpg', cv.IMREAD_COLOR)

im = cv.resize(im, (512,512))
im_db = cv.resize(im_db, (512,512))

# Remove salt & pepper noise and convert image to grayscale
filtered_im_q = cv.medianBlur(im,3)
filtered_im_db = cv.medianBlur(im_db,3)

img_gray_q = cv.cvtColor(filtered_im_q,cv.COLOR_BGR2GRAY)
img_gray_db = cv.cvtColor(filtered_im_db,cv.COLOR_BGR2GRAY)

# Compute background and text masks
bg_mask_q,_ = compute_mask(im, "try_mask1", 'qsd1w4')
text_mask_q = find_text(img_gray_q, bg_mask_q, "try_text1")

# Call SIFT creator
sift = cv.xfeatures2d.SIFT_create()

# Compute SIFT keypoints and descriptor
kp1, des1 = compute_SIFT_kp_and_des(img_gray_q, bg_mask_q[0], text_mask_q[0], sift)
kp2, des2 = compute_SIFT_kp_and_des(img_gray_db, None, None, sift)

# Variable to store descriptors
descriptors=[]
descriptors.append(des1)
descriptors.append(des2)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptors[0],descriptors[1],k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
matches_good = 0

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        matches_good += 1

# Number of "true" matches
print(matches_good)

# Matches drawing
draw_params = dict(matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = cv.DrawMatchesFlags_DEFAULT)

# Plot matches
img3 = cv.drawMatchesKnn(img_gray_q,kp1,img_gray_db,kp2,matches,None,**draw_params)
plt.imshow(img3,)
plt.show()
"""

#########

"""
print(descriptor1)
print(descriptor2)
# Final image with drawn keypoints
img1 = cv.drawKeypoints(img_gray_q, kp1, im)
img2 = cv.drawKeypoints(img_gray_db, kp2, im_db)
#img = cv.drawKeypoints(img_gray,kp,im,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

"""
"""
# Show keypoints
cv.imshow("image", img1)
cv.waitKey(0)
cv.destroyAllWindows()

# Show keypoints
cv.imshow("image", img2)
cv.waitKey(0)
cv.destroyAllWindows()
"""

"""
# Construct BFMatcher object with default parameters
bf = cv.BFMatcher()
matches = bf.knnMatch(descriptor1, descriptor2, k = 2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# Draw matches (list of lists)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()

print(len(good))
"""