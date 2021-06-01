import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt

#todo: 
#find out how to add multiple subfolder to the searching path!
#=> Tools => Python Path Manager => Add Path

#%% display Image
def showMyImage(windowName, img):
    height = img.shape[0]
    width = img.shape[1]
    cv.namedWindow(windowName, cv.WINDOW_NORMAL) #needed to resize the window
    cv.imshow(windowName, img)
    cv.imshow(windowName, img)
    cv.resizeWindow('windowName', (int(width*0.1), int(height*0.1)))
    k = cv.waitKey(0)
    if k == ord("q"):
        cv.destroyAllWindows()
        print('All windows destroyed.')

#%% 1) Aquire the image

# There are 3 variants to aquire the image:
    # first: read image
    # second: read camera data, press button s to choose the current frame
    # third: read video data, press button s to choose the current frame
    
    #the image for the imagepreprocessing should be saved
    #otherwise it will take the last saved image (if there was one)

variantImageAcquisition = 1
saveName = "testSave1.png"
saveCameraVideoName = "testVideo1.avi"
desire2SaveCameraVideo = 0 #if you want to save a video from the camera data put 1 else 0

if (variantImageAcquisition == 1): #read image only
    img = cv.imread(cv.samples.findFile("lego-rot0-1a.jpg"))
    if img is None:
        sys.exit("Could not read the image.")
    height = img.shape[0]
    width = img.shape[1]
    cv.namedWindow('windowName', cv.WINDOW_NORMAL) #needed to resize the window
    cv.imshow("windowName", img)
    cv.resizeWindow('windowName', (int(width*0.1), int(height*0.1)))
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite(saveName, img)
        print('Image with name: ',saveName, ' was saved!')
        cv.destroyAllWindows()
        print('All windows destroyed.')
    elif k == ord("q"):
        cv.destroyAllWindows()
        print('All windows destroyed.')
        
        
elif (variantImageAcquisition == 2):  #read camera data   
    cap = cv.VideoCapture(0)
    
    #Define the codec and create VideoWriter object
    if desire2SaveCameraVideo:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(saveCameraVideoName, fourcc, 20.0, (640,  480))
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        # if frame is read correctly success is True
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv.imshow('frame', frame)
        if desire2SaveCameraVideo:
            out.write(frame)
        
        k = cv.waitKey(1)
        if k == ord('q'):
            print('"q" was pressed! Process aborted!')
            break
        elif k == ord('s'):
            cv.imwrite(saveName, frame)
            print('"s" was pressed! Image with name: ',saveName, ' was saved!')
            break
    # When everything done, release the capture
    cap.release()
    if desire2SaveCameraVideo:
        out.release()
        print('Video with name: ',saveCameraVideoName, ' was saved!')

    cv.destroyAllWindows()
    print('All windows destroyed.')
    

elif (variantImageAcquisition == 3): #read video
    cap = cv.VideoCapture('bird.avi')
    while cap.isOpened():
        success, frame = cap.read()
        # if frame is read correctly success is True
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv.imshow('frame', frame)
        k = cv.waitKey(1)
        if k == ord('q'):
            print('"q" was pressed! Process aborted!')
            break
        elif k == ord('s'):
            cv.imwrite(saveName, frame)
            print('"s" was pressed! Image with name: ',saveName, ' was saved!')
            break

    cap.release()
    cv.destroyAllWindows()
    print('All windows destroyed.')
    

#%% 2) Preprocessing the image

#uses the image out of section "1) Aquire the image"
doPreProcess = 1
if doPreProcess:
    aqImg = cv.imread(cv.samples.findFile(saveName))
    if aqImg is None:
        sys.exit("Could not read the image.")
        
    height = aqImg.shape[0]
    width = aqImg.shape[1]
    
    
    #blur = cv.blur(img,(25,25))
    #blur = cv.GaussianBlur(aqImg,(25,25),0)
    hsvImage = cv.cvtColor(aqImg, cv.COLOR_BGR2HSV)
    grayImage = cv.cvtColor(aqImg, cv.COLOR_BGR2GRAY)
    #h = hue: color itself: {0-180}; s = saturation: colorfullness of the color: {0-255}; v = value: {0-255}: how light or dark is the color
    

    
    """
    lower = np.array([0,0,20])
    upper = np.array([30,30,255])
    #lower = np.array([80,50,50])
    #upper = np.array([200,255,255])
    mask0 = cv.inRange(hsvImage,lower,upper)"""
    
    variantPre = 2
    if variantPre == 1:
        
        medianImage = cv.medianBlur(grayImage, 9)
        blurAndMedianImage = cv.blur(medianImage,(9,9))
        
    elif variantPre == 2:
        im2 = cv.blur(grayImage,(5,5))
        blurAndMedianImage = cv.medianBlur(im2, 5)
    #mask = cv.inRange(medianImage,200,255)
    
    edges = cv.Canny(blurAndMedianImage,10, 50)
    
    showMyImage('1', edges)

    """plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()"""
    
    
    
    """contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    showMyImage('test', mask)
    showMyImage('test', grayImage)
    showMyImage('test', medianImage)
    showMyImage('test', blurAndMedianImage)"""


