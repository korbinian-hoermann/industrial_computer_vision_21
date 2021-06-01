#%% Imported Libraries
import cv2
import numpy as np
import scipy.spatial as sp

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
    #%% Preprocessing of the image

    im = cv2.imread(cv2.samples.findFile("testNewEnvironment.jpg"))
    im1 = ResizeWithAspectRatio(im, width=500)
    im2 = cv2.bilateralFilter(im1,25,75,75) 
    im3 = cv2.medianBlur(im2, 5)
    im4 = cv2.cvtColor(im3, cv2.COLOR_BGR2Lab)
    
    #%% Color-Calibration
    
    main_colors = [] 
    dictColor = ['red','green','orange','darkgreen','blue','purple','white','gray','yellow','background']
    for i in range(0, len(dictColor)):
    
        # Select ROI for calibrate the color
        r = cv2.selectROI(img = im1,windowName = dictColor[i])
        
        # Crop the selected roi
        imCrop = im4[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        
        # Display cropped image in LAB-Space
        """cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        
        l, a, b = cv2.split(imCrop)

        meanL = round(np.mean(l),2)
        meanA = round(np.mean(a),2)
        meanB = round(np.mean(b),2)

        main_colors.append((meanL, meanA, meanB))  

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    #%% Nearest-Neighbor-Classification
    h,w,bpp = np.shape(im4)
    classImage = np.zeros(shape=(h,w,1))
    for py in range(0,h):
        for px in range(0,w):
            input_color = (im4[py][px][0],im4[py][px][1],im4[py][px][2])
            tree = sp.KDTree(main_colors) 
            distance, result = tree.query(input_color) 
            #nearest_color = main_colors[result]
            classImage[py][px] = result
    
    #%% Show binary image for every calibrated color
    for i in range(0, len(dictColor)):
        tempImage = np.zeros(shape=(h,w,1))
        tempImage[classImage == i] = 1
        
        #use morphological opening to get rid of false positive
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opened_mask = cv2.morphologyEx(tempImage, cv2.MORPH_OPEN, kernel)
        
        #masked_img = cv2.bitwise_and(img, img, mask=opened_mask)
        cv2.namedWindow(dictColor[i] , cv2.WINDOW_NORMAL) #needed to resize the window
        cv2.imshow(dictColor[i] , opened_mask)
        cv2.resizeWindow(dictColor[i] , (int(w*0.5), int(h*0.5)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()