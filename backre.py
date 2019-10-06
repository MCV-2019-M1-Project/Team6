#IMPORTS
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os


#Parameters
COLORSPACE = cv.COLOR_BGR2HSV

#Read QSD2
images = glob.glob('./qsd2_w1/*.jpg')
print("Query set 2 has " + str(len(images))+" images")

for f in sorted(images):
    
    #Numeric identifier for every image
    name=os.path.splitext(os.path.split(f)[1])[0]
    print("Image name:"+str(name))
    
    #Read ground truth mask
    g_t=cv.imread('qsd2_w1/'+name+'.png',cv.IMREAD_COLOR)
    g_t=img=cv.cvtColor(g_t,cv.COLOR_BGR2GRAY)

    #Read image and conver it to the chosen color space
    img=cv.imread(f,cv.IMREAD_COLOR)
    img=cv.cvtColor(img,COLORSPACE)
    
    #Compute channel on the chosen color space
    num_ch=1
    ch=img[:,:,num_ch]
  
    #Height and width of channel (=image dims)
    height,width=ch.shape[:2]
    
    #Amount of pixels per each side of the image
    percent=0.03
    aop_h=int(round(percent*height))
    aop_w=int(round(percent*width))
    
    #Crop different portions of the image containing the background
    portion=ch[0:aop_h, 0:width]
    portion2=ch[0:height,0:aop_w]
    portion3=ch[0:height,width-aop_w:width]
    portion4=ch[height-aop_h:height,0:width]
     
    #Compute thresholds from chosen image sides
    min_p1=int(np.amin(portion))
    max_p1=int(np.amax(portion))
    
    min_p2=int(np.amin(portion2))
    max_p2=int(np.amax(portion2))

    min_p3=int(np.amin(portion3))
    max_p3=int(np.amax(portion3))
    
    min_p4=int(np.amin(portion4))
    max_p4=int(np.amax(portion4))

    #Compute absolute thresholds from chosen portions
    minval=min(min_p1,min_p2)
    maxval=min(max_p1,max_p2)
    
    mask=np.zeros((height,width))
    
    cv.imwrite('newmasks/'+name+'.png',mask)
    
    #Loop over the channel and create the mask
    for i in xrange(ch.shape[0]):
        for j in xrange(ch.shape[1]):
            if ch[i,j]>=minval and ch[i,j]<=maxval:
                #Pixels belong to the background
                mask[i,j]=0
            else:
                #Pixels belong to the painting
                mask[i,j]=255
    
    #Save binary mask
    cv.imwrite('masks/'+name+'.png',mask)
    

# =============================================================================
#     #Compute evaluation metrics    
#     pixelTP,pixelFP,pixelFN,pixelTN = performance_accumulation_pixel(mask,g_t)
#     pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
# 
#     print("Precision: "+str(pixel_precision))
#     print("Accuracy: "+str(pixel_accuracy))
#     print("Specificity: "+str(pixel_specificity))
#     print("Recall (sensitivity): "+str(pixel_sensitivity))
# =============================================================================
