#%% Imported Libraries
import cv2
import numpy as np
import random as rng

#%% Functions
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

if __name__ == '__main__' :
    #%% Preprocessing the image
    
    im = cv2.imread(cv2.samples.findFile("lego-rot0-1a.jpg"))
    im1 = ResizeWithAspectRatio(im, width=500)

    withPreProcessing = 1
    if withPreProcessing:
        im2 = cv2.bilateralFilter(im1,25,75,75) #best!
        imF = cv2.medianBlur(im2, 5)
        cv2.namedWindow('afterPreProcess')
        cv2.imshow('afterPreProcess', imF)
    else:
        imF = im1

    cv2.namedWindow('rawImage')
    cv2.imshow('rawImage', im1)

    imgGray = cv2.cvtColor(imF, cv2.COLOR_BGR2GRAY)
    
    #%% Canny-Edgedetection + Morphological Post-processing
    
    # get values with script "edgeDetectionInteractive.py"
    v1 = 0 #25
    v2 = 60 #60
    
    edgesRaw = cv2.Canny(imgGray,v1,v2)
    
    #dilation to connect edge-pieces that are naturally not connected
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edgesAfterDilation = cv2.dilate(edgesRaw,kernel,iterations = 1)
    #thinning to get rid of additional pixels from dilation
    edgesAfterDilAndThin = cv2.ximgproc.thinning(edgesAfterDilation)
    
    cv2.namedWindow('edgesRaw')
    cv2.imshow('edgesRaw',edgesRaw)
    
    cv2.namedWindow('edgesAfterDilation')
    cv2.imshow('edgesAfterDilation',edgesAfterDilation)
    
    cv2.namedWindow('edgesAfterDilAndThin')
    cv2.imshow('edgesAfterDilAndThin',edgesAfterDilAndThin)
    
    #%% Extract Contours
    
    contours, hierarchy = cv2.findContours(edgesAfterDilAndThin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    drawing = np.zeros((edgesRaw.shape[0], edgesRaw.shape[1], 3), dtype=np.uint8)
    
    for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            if cv2.contourArea(contours[i]) > 50:
                cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        
    # Show in a window
    cv2.namedWindow('Contours')
    cv2.imshow('Contours', drawing)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
