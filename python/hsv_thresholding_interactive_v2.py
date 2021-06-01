####################  Imported Libraries  ####################
import cv2
import numpy as np
import time


####################  Variables  ####################
IMAGE_FOR_CALIBRATION = "../hsv_thresholds/images_for_color_calibration/image_0.png"


####################  Functions  ####################

def nothing(x):
    pass

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

def saveHSVvalues(color, l_h, u_h, l_s, u_s, l_v, u_v):
    baseName = 'Thresholds.txt'
    filename = color + baseName
    file1 = open(filename,"w")
    L = str(l_h) + ' ' + str(u_h) + ' ' + str(l_s) + ' ' + str(u_s) + ' ' + str(l_v) + ' ' + str(u_v)
    file1.writelines(L)
    file1.close() #to change file access modes
    outputString = 'File with name ' + color +baseName + ' was written.'
    print(outputString)
    

if __name__ == '__main__' :
    #%% Create Trackbar
    
    #define Colors
    dictColor = ['red','green','lightgreen','orange','purple','brown','gray','blue','yellow','lightbrown']
    #dictColor = ['red','green']
    cv2.namedWindow("Tracking")
    cv2.namedWindow("Saving", cv2.WINDOW_NORMAL)
    
    
    cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
    
    cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
    
    cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
    
    for i in range(0, len(dictColor)):
        cv2.createTrackbar(dictColor[i], "Saving", 0, 1, nothing)


    #%% Image Preprocessing
    
    im = cv2.imread(IMAGE_FOR_CALIBRATION)
    im1 = ResizeWithAspectRatio(im, width=500)
    im2 = cv2.bilateralFilter(im1,25,75,75) 
    im3 = cv2.medianBlur(im2, 5)
    
    hsvMain = cv2.cvtColor(im3, cv2.COLOR_BGR2HSV)

    #%% Adjust HSV-Values interactively with Trackbar
    
    #intialise all bits to 0
    bitSet = dict.fromkeys(dictColor,0)
    
    while True:
        hsv = hsvMain.copy()
        
        # H-thresholds
        l_h = cv2.getTrackbarPos("LH", "Tracking")
        u_h = cv2.getTrackbarPos("UH", "Tracking")
        
        # S-thresholds
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        
        # V-thresholds
        l_v = cv2.getTrackbarPos("LV", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")
        
        for i in range(0, len(dictColor)):
            currentColor = dictColor[i]
            currentBit = cv2.getTrackbarPos(currentColor, "Saving")
            
            if currentBit == 1 & ~bitSet[currentColor]:
                bitSet[currentColor] = 1
                saveHSVvalues(currentColor, l_h, u_h, l_s, u_s, l_v, u_v)


        # summary thresholds
        l_b = np.array([l_h, l_s, l_v]) #lower values
        u_b = np.array([u_h, u_s, u_v]) #upper values
        
        mask = cv2.inRange(hsv, l_b, u_b)
        res = cv2.bitwise_and(im3, im3, mask=mask)
    
        #merge 2 images to have less windows
        img_concate_Hori1 = np.concatenate((im3,res),axis=1)

        cv2.imshow("Tracking",mask)
        cv2.imshow("Tracking2",img_concate_Hori1)
    
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    cv2.destroyAllWindows()
    
