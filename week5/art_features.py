import cv2 as cv
import numpy as np
import math
import colorgram
import tamura
import pywt
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def art_features(img, image_name):
    
    img = cv.resize(img, (round(img.shape[1]/4),round(img.shape[0]/4)))
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    npx = height*width
    
    #Mean saturation and brightness
    saturation = np.sum(img_hsv[:][:][1])/npx
    brightness = np.sum(img_hsv[:][:][2])/npx

    #Pleasure, arousal and dominance
    pleasure = 0.69*brightness +0.22*saturation 
    arousal = -0.31*brightness +0.60*saturation 
    dominance = 0.76*brightness +0.32*saturation

    #Mean hue and dispersion, with and without saturation
    a = 0
    a_s = 0
    b = 0
    b_s = 0
    for x in range(height):
        for y in range(width):
            a += math.cos(img_hsv[x][y][1])
            a_s += img_hsv[x][y][1]* math.cos(img_hsv[x][y][0])
            b += math.sin(img_hsv[x][y][1])
            b_s += img_hsv[x][y][1] * math.sin(img_hsv[x][y][0])
    
    mean_hue = math.atan2(b,a)
    hue_variance = 1 - math.sqrt(a*a+b*b)/npx
    mean_hue_saturation = math.atan2(b_s,a_s)
    hue_variance_saturation = 1 - math.sqrt(a_s*a_s+b_s*b_s)/(saturation*npx)

    #Colorfulness
    img_signature = np.zeros((64,4), np.float32)
    uniform_signature = np.zeros((64,4), np.float32)
    hist = cv.calcHist([img], [0, 1, 2], None, [4, 4, 4], [0, 255, 0, 255, 0, 255])/npx 
    i=0
    for x in range(0,4):
        for y in range(0,4):
            for z in range(0,4):
                img_signature[i][0] = hist[x][y][z]
                uniform_signature[i][0] = 1/64
                img_signature[i][1] = x
                uniform_signature[i][1] = x
                img_signature[i][2] = y
                uniform_signature[i][2] = y
                img_signature[i][3] = z
                uniform_signature[i][3] = z
                i+=1
    
    colorfulness = cv.EMD(uniform_signature, img_signature, cv.DIST_L2)[0]

    #Main colours
    main_colors = []
    colors = colorgram.extract(image_name, 3)
    for i in range(0,3):
        for j in range(0,3):
            main_colors.append(colors[i].hsl[j])    
        main_colors.append(colors[i].proportion)

    #Tamura texture features: coarseness, contrast and directionality
    coarseness = tamura.coarseness(img_gray, 3)
    contrast = tamura.contrast(img_gray)
    directionality = tamura.directionality(img_gray)

    #Wavelet texture descriptors
    wavelets = []
    coeffs_h= pywt.wavedec2(img_hsv[:][:][0], 'db1', level=3)
    coeffs_s= pywt.wavedec2(img_hsv[:][:][1], 'db1', level=3)
    coeffs_v= pywt.wavedec2(img_hsv[:][:][2], 'db1', level=3)
    for i in range(1,4):
        sum_h = 0
        sum_s = 0
        sum_v = 0
        mod_h = 0
        mod_s = 0
        mod_v = 0
        for j in range(0,3):
            sum_h += np.sum(coeffs_h[i][j])
            sum_s += np.sum(coeffs_s[i][j])
            sum_v += np.sum(coeffs_v[i][j])
            mod_h += math.sqrt(np.sum(coeffs_h[i][j][1]**2))
            mod_s += math.sqrt(np.sum(coeffs_s[i][j][1]**2))
            mod_v += math.sqrt(np.sum(coeffs_v[i][j][1]**2))
        wavelets.append(sum_h/mod_h)
        wavelets.append(sum_s/mod_s)
        wavelets.append(sum_v/mod_v)
    
    #Level of detail (number of regions after segmentation)
    segments_fz1 = felzenszwalb(img, scale=100, sigma=0.8, min_size=30)
    segments_fz2 = felzenszwalb(img, scale=300, sigma=0.8, min_size=30)
    segments_fz3 = felzenszwalb(img, scale=500, sigma=0.8, min_size=30)
    
    lod = [len(np.unique(segments_fz1)), len(np.unique(segments_fz2)), len(np.unique(segments_fz3))]

    # plt.imshow(mark_boundaries(img, segments_fz1))
    # plt.waitforbuttonpress()

    #Dynamics and number of lines
    edges = cv.Canny(img_gray,50,100)
    lines = cv.HoughLines(edges,1,np.pi/180,100)
    vert = 0
    horz = 0
    tilt = 0
    if np.any(lines != None):
        for line in lines:
            for rho, theta in line:
                if np.sin(theta) > -0.1 and np.sin(theta) < 0.1:
                    vert += 1
                elif np.sin(theta-np.pi/2.0) > -0.1 and np.sin(theta-np.pi/2.0) < 0.1:
                    horz +=1
                else:
                    tilt +=1
        num_lines = len(lines)
        vert /= num_lines
        horz /= num_lines
        tilt /= num_lines
    else:
        num_lines = 0

  
    dynamics = [num_lines, vert, horz, tilt]


    #Colour features: saturation + brightness + pleasure + arousal + dominance + mean_hue + hue_variance + mean_hue_saturation + hue_variance_saturation + colorfulness + main_colors
    #Texture features: coarseness + contrast + directionality
    #Composition features: lod dynamics

image_name = '../database/bbdd_00100.jpg'
img = cv.imread(image_name, cv.IMREAD_COLOR)
art_features(img, image_name)