# industrial_computer_vision_21
Repo for the course "Industrial Computer Vision" at University of Aveiro, group 1

The work for this project was done with Python 3.7 using the following frameworks and libraries: 
- opencv
- tensorflow
- keras
- numpy
- tkinter

Some help for the installation of those libraries:
Libraries to install: 
openCV 4 (Tutorial: https://robu.in/installing-opencv-using-cmake-in-raspberry-pi/)

And additionally: 
sudo apt-get install python3-opencv
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev 
sudo apt-get install libqtgui4 
sudo apt-get install libqt4-test
From <https://www.raspberrypi.org/forums/viewtopic.php?t=232294> 


Pip3 install Imutils
Pip2 install tkinter

Tensorflow 
Pip3 install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_armv7l.whl

From <https://towardsdatascience.com/3-ways-to-install-tensorflow-2-on-raspberry-pi-fe1fa2da9104> 


Tensorflow lite
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl


## Table of Contents  
- [ignore](#ignore)
  - [PreProcessAndSegmentation](#PreProcessAndSegmentation)
  - [OpenCV_tutorials](#OpenCV_tutorials)
- [hsv_thresholds](#hsv_thresholds)
- [models](#models)
- [own_dataset](#own_dataset)
- [python](#python)



<a name="ignore"/>
<a name="PreProcessAndSegmentation"/>
<a name="OpenCV_tutorials"/>
<a name="hsv_thresholds"/>
<a name="models"/>
<a name="own_dataset"/>
<a name="python"/>


## ignore
This folder contains outdated scripts and code snippets and can be ignored in the greater context.
Anyway, an important step in the pipeline is the preprocessing of obtained images. Codes for this process can be found in the folder

- PreProcessAndSegmentation

Scripts | Description
-------|-----------------------------------
aquireImage.py   | aquire image via 3 options: a) load image b) save frame of camera c) save frame of a video
edgeDet.py   | Script for canny-edge-detection with (pre- and postprocessing)
edgeDetectionInteractive.py   | Interactive script to find out the 2 parameters for canny-edge-detection (with preprocessing)
hsvThresholdingInteractive.py   | Interactive script to find out colorbased thresholding in HSV-Colorspace
nearest-neighbor-classify-LAB.py   | Color-based calibration and segmentation in LAB-Space (Nearest Neighbor-Method)

- OpenCV_tutorials

Useful code snippets and notes of some the opencv tutorials.

## hsv_thresholds
This folder contains .txt files which contain lower and upper boundaries of the corresponfing HSV Color and another folder, in which images are save on which the color calibration can be performed.

## calibration
This folder contains .jpg files which were used to do the intrinsic calibration. Furthermore, the script to perform the calibration is also included.
At the end of the script the mean-error is calculated over all images used for the intrinsic calibration.
The intrinsic parameters are stored in a json file. 
The immediate following reading of the parameters is good for comparing if the writing worked, so that when the string is split into its variables again, it can be compared if everything is correct.

The checkerboard pattern from the OpenCV page that was printed to perform the calibration is also added.
Moreover, the intrinsic parameters stored in the JSON file are also uploaded.

## models
This folder contains .tflite models which were trained on the dataset and can be used to perform predictions on the RaspberryPi.
Therefore the image data set was zipped (.zip) and uploaded to Google Drive. In the next step Google GPU Backend for Google Colab was used for training the CNN faster in the cloud.
The trained model is then saved on Google Drive, downloaded and transfered to the RaspberryPi where it can be used to perform predictions on new image data.

Notebook: https://drive.google.com/file/d/1PNyl5kIR6DC3Wm9g0pjw5vuS2nyDFgEM/view?usp=sharing 

## own_dataset
This is a self created dataset of lego bricks. 
![](images/data.png) 

Currently it consists of **5 different classes** of Lego and contains **~1100 images (".png", RGB encoded, 224x224)**.


## python
This folder contains 3 the python scipts which can be run on the RaspberryPi:

- **hsv_thresholding_interactive_v2.py**

Description | Example image
-------|-----------------------------------
This script opens 3 windows and allows to set different thresholds in the HSV colorspace. Those thresholds can than be saved to .txt files.   | ![](images/hsv_thresholding_interactive.jpg) 



- **image_accuisition.py**

Description | Example image
-------|-----------------------------------
This script asks the user for which class of lego he wants to create images.   | ![](images/acquire_images_console.jpg)
After the input of the prefered id, the program shows a video stream of the camera and in a second window the detected video. By moving the camera, new ids are given to the pieces, showing them in different lightning and angles.   |  ![](images/acquire_images_detection.jpg)
After hitting 'ESC', two windows are shown. One contains information about all the detected pieces, the second one only shows the Id and the extracted image of this piece. By clicking on those columns, the images are saved to the 'dataset/<id>' folder.   |  ![](images/overview_and_save.jpg)

- **lego_brick_detection.py**
  
 Description | Example image
-------|-----------------------------------
This script detects lego pieces in the video stream and uses a tflite model which was trained on Google Colab to predict the class of the lego pieces   | ![](images/detection_and_classification.jpg)
  After hitting 'ESC', two windows are shown. One contains information about all the detected pieces, the second one only shows the Id, the extracted image and the predicted class of this piece. In case this predicted id is correct, those columns can be clicked and the images are saved to the 'dataset/<id>' folder in order to increase the size of the dataset with wich the model can be trained.   |  ![](images/final_overview.jpg)

