#%% Imported Libraries
import cv2
import numpy as np

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

def doEdges():
    edges = cv2.Canny(imGray,v1,v2)
    edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    res = np.concatenate((im1,edges),axis = 1)
    cv2.imshow('Result',res)

def setVal1(val):
    global v1
    v1 = val
    doEdges()
    
def setVal2(val):
    global v2
    v2 = val
    doEdges()

if __name__ == '__main__' :
    #%% Preprocessing of the image

    im = cv2.imread(cv2.samples.findFile("lego-rot0-1a.jpg"))
    im1 = ResizeWithAspectRatio(im, width=500)
    im2 = cv2.bilateralFilter(im1,25,75,75) 
    im3 = cv2.medianBlur(im2, 5)
    imGray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    
    #%% Adjust Canny-Thresholds interactively with Trackbar

    #initial values
    v1 = 0
    v2 = 0
    
    #create trackbar
    cv2.namedWindow('Result') #IMPORTANT: namedWindow before createTrackbar
    cv2.createTrackbar('Val1','Result',0,200,setVal1)
    cv2.createTrackbar('Val2','Result',0,200,setVal2)
    
    initIm = np.concatenate((im1,im1),axis = 1)
    cv2.imshow('Result',initIm)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    