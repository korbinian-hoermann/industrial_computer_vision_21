#%% Variable Description (as reminder)

"""
objectPoints	A vector of vector of 3D points.The outer vector contains as many elements as the number of the pattern views.
imagePoints	A vector of vectors of the 2D image points.
imageSize	Size of the image
cameraMatrix	Intrinsic camera matrix
distCoeffs	Lens distortion coefficients.
rvecs	Rotation specified as a 3×1 vector. The direction of the vector specifies the axis of rotation and the magnitude of the vector specifies the angle of rotation.
tvecs	3×1 Translation vector.
"""
#%% Import Libraries
import numpy as np
import cv2 as cv
import glob
import json

#%% Camera Calibration 

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#the folder should only contain images for the calibration process
images = glob.glob('C:/Users/Pascal/.spyder-py3/openCV_Lego/calib_pi/**' ) 
counter = 0 
for fname in images:
	
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        counter = counter + 1
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(50)
cv.destroyAllWindows()

#%% Get Intrinsic Parameters
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#%% Undistortion (Example)
desireUndistortImage = 0 #undistort the last image (as example)
if desireUndistortImage:
	h,  w = img.shape[:2]
	newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
	
	# undistort
	dst = cv.undistort(img, mtx, dist, None, newcameramtx)
	# crop the image
	x, y, w, h = roi
	dst = dst[y:y+h, x:x+w]
	cv.imwrite('undistortedImageExample.png', dst)

#%% Calculate mean error (off all used images)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

#%% Writing data to JSON-File
mtx2json = mtx.tolist()
dist_coeff2json = dist.tolist()
data2json = {"camera_matrix": mtx2json, "dist_coeff": dist_coeff2json}
fname = "cameraIntrinsicLegoProject.json"

file = open(fname,"w")
json.dump(data2json, file)
file.close()

#%% Reading data from JSON-File
file = open(fname, 'r')
dataJson = json.load(file)
file.close()
  
mtx1 = np.array( dataJson['camera_matrix'] )
dist1 = np.array( dataJson['dist_coeff'] )

