import cv2
import numpy as np
import random as rng
import imutils
from PIL import Image
from tkinter import *
from PIL import Image, ImageTk
import time
from datetime import datetime
import tflite_runtime.interpreter as tflite
import tensorflow as tf


##############################  Variables  #############################

# Path of trained tflite model for prediction
TFLITE_FILE_PATH = "/home/pi/Desktop/camera/models/custom_cnn_own_data_DA_v4.tflite"

# Dictionary for mapping of class ids to string representations
classes_dict = {
    0: "roof tile 2x2",
    1: "plate 2x4",
    2: "plate 1x2",
    3: "A 2x4",
    4: "U 2x3"
}

# Dictionary of colors which should be detected
dictColor = ['yellow', 'green', 'orange', 'red', 'blue']



##############################  Classes  ###############################
class Lego: 
  """This class will be used as our data structure to safe information about detected lego."""
  def __init__(self, id, color, position, rgb_image, aspect_ratio, area, legoclass, tk_image):
	  self.id = id #int
	  self.legoclass = legoclass
	  self.color = color #string
	  self.position = position #[x,y] of middlepoint (from findcontours)
	  self.rgb_image = rgb_image #3dim array/matrix
	  self.tk_image = tk_image #3dim array/matrix
	  self.aspect_ratio = round(aspect_ratio,2) #float
	  self.area = area #int - number of pixels
	  self.identification = get_class_from_class_id(int(self.legoclass)) #string: ex: '6x3 brick'
	  
  def describe(self):
	  """Returns a tuple describing the lego piece 
	  (<info as string>, tk_image)
	  """
	  
	  return (f"{self.id:<15}{self.legoclass:<15}{self.identification:<20}{self.color:<20}{str(self.position):<20}{self.aspect_ratio:<20}{self.area:<20}", self.tk_image)
	  
class VerticalScrolledFrame(Frame):
	"""A pure Tkinter scrollable frame

	* Use the 'interior' attribute to place widgets inside the scrollable frame
	* Construct and pack/place/grid normally
	* This frame only allows vertical scrolling
	"""
	def __init__(self, parent, *args, **kw):
		Frame.__init__(self, parent, *args, **kw)			

		# create a canvas object and a vertical scrollbar for scrolling it
		vscrollbar = Scrollbar(self, orient=VERTICAL)
		vscrollbar.pack(fill=Y, side=RIGHT, expand=TRUE)
		canvas = Canvas(self, height=1000, bd=0, highlightthickness=0,
						yscrollcommand=vscrollbar.set)
		canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
		vscrollbar.config(command=canvas.yview)

		# reset the view
		canvas.xview_moveto(0)
		canvas.yview_moveto(0)

		# create a frame inside the canvas which will be scrolled with it
		self.interior = interior = Frame(canvas)
		interior_id = canvas.create_window(0, 0, window=interior,
										   anchor=NW)

		# track changes to the canvas and frame width and sync them,
		# also updating the scrollbar
		def _configure_interior(event):
			# update the scrollbars to match the size of the inner frame
			size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
			canvas.config(scrollregion="0 0 %s %s" % size)
			if interior.winfo_reqwidth() != canvas.winfo_width():
				# update the canvas's width to fit the inner frame
				canvas.config(width=interior.winfo_reqwidth())

		interior.bind('<Configure>', _configure_interior)

		def _configure_canvas(event):
			if interior.winfo_reqwidth() != canvas.winfo_width():
				# update the inner frame's width to fill the canvas
				canvas.itemconfigure(interior_id, width=canvas.winfo_width())
		canvas.bind('<Configure>', _configure_canvas)
  

##############################  Functions  #############################

def to_bgr(img):
	sub = img.convert("RGBA")
	data = np.array(sub) 
	red, green, blue, alpha = data.T 
	data = np.array([blue, green, red, alpha])
	data = data.transpose()
	sub = Image.fromarray(data)
	return sub


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


def pil_to_cv(pil_img):
	"""This function converts an image from the PIL format to 
	the openCv format."""
	
	return np.asarray(pil_img)


def expand2square(cv_img, background_color):
	img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
	pil_img = Image.fromarray(img)
   
	width, height = pil_img.size
	if width == height:
		return pil_to_cv(pil_img)
	elif width > height:
		result = Image.new(pil_img.mode, (width, width), background_color)
		result.paste(pil_img, (0, (width - height) // 2))
		return pil_to_cv(result)
	else:
		result = Image.new(pil_img.mode, (height, height), background_color)
		result.paste(pil_img, ((height - width) // 2, 0))
		return pil_to_cv(result)
	
def resize_image_with_padding(img):
	'''This function rescales a given image to a square of 
	the size 224x224 by surrounding it with a black padding.
	'''
	
	img_with_padding= expand2square(img, (0, 0, 0))
	img_reshaped = img_with_padding.copy()
	img_reshaped = cv2.resize(img_with_padding, (224, 224))
	img_rgb = cv2.cvtColor(img_reshaped, cv2.COLOR_BGR2RGB)
	
	return img_rgb

def decode_img(img):
  img = tf.image.decode_png(img, channels=3) # channels 3 -> rgb
  img = tf.image.convert_image_dtype(img, tf.float32)
  return tf.image.resize(img, [224, 224])

def predict_class(rgb_img):
	"""This function predicts the class of lego for a given rgb image"""
	#Preprocess the image to required size and cast	
	img_tensor = tf.convert_to_tensor(rgb_img) 
	img_gray = tf.image.rgb_to_grayscale(img_tensor)
	img = tf.image.convert_image_dtype(img_gray, tf.float32)
	
	input_data = np.array(tf.expand_dims(img, 0), dtype=np.float32)

		
	#set the tensor to point to the input data to be inferred
	input_index = interpreter.get_input_details()[0]["index"]
	interpreter.set_tensor(input_index, input_data)#Run the inference
	interpreter.invoke()
	
	output_details = interpreter.get_output_details()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	results = np.squeeze(output_data)
	
	return int(np.argmax(results))
	
def get_class_from_class_id(class_id):
	"""This function returns the string representation of a 
	lego class id"""
	
	return classes_dict[class_id]


######################  Setup of prediction model ######################
# Load CNN model (TFLite)
# Load the TFLite model in TFLite Interpreter
interpreter = tflite.Interpreter(model_path=TFLITE_FILE_PATH)

# Allocate the tensors
interpreter.allocate_tensors()


############################  Camera setup  ############################
cam = cv2.VideoCapture(0)
cv2.namedWindow("Original video")

img_counter = 0
dictLegoBricks = {}


####################  Preparations for Tkinter GUI #####################
root = Tk()
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
w = 900
h = 400
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)
root.title("Overview - extracted information")
root.geometry('%dx%d+%d+%d' % (w, h, x, y))
root.update()
canvas = Canvas(root, bg="gray70", width=w, height=h)
canvas.pack()


time.sleep(.2)
ret, frame = cam.read()
time.sleep(.5)
frame_cnt = 0

frame_rate = 15
prev = 0

while True:
	
	time_elapsed = time.time() -prev
	
	# Get frame from camera
	ret, frame = cam.read()
	if not ret: 
		print("failed to grab frame")
		break
	
	# Ignnore the first 50 frames (Let camera adjust to light)
	frame_cnt += 1
	if frame_cnt < 50:
		continue
		
	if time_elapsed > 1./frame_rate:
		prev = time.time()
		
		# Preprocess image
		im1 = ResizeWithAspectRatio(frame, width=500)
		im2 = cv2.bilateralFilter(im1,25,75,75) 
		preprocessed_frame = cv2.medianBlur(im2, 5)
		
		# Create binary masks via HSV
		hsvMain = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2HSV)
		
		hsvLow = dict.fromkeys(dictColor)
		hsvHigh = dict.fromkeys(dictColor)
		hsvHigh = dict.fromkeys(dictColor)
		maskColl = dict.fromkeys(dictColor)
		maskWithColorColl = dict.fromkeys(dictColor)

		# Read all the hsv-thresholds out of the textfiles
		for i in range(0, len(dictColor)):
			
			hsv = hsvMain.copy()
			
			currentColor = dictColor[i]
			baseName = 'Thresholds.txt'
			filename = "../hsv_thresholds/" + currentColor + baseName
			file1 = open(filename,"r")
			rawRead = file1.readlines()
			a = rawRead[0]
			fullSeq = a.split(' ')
			
			# Lower boundaries
			hsvLow[currentColor] = [ int(float(fullSeq[0])),int(float(fullSeq[2])),int(float(fullSeq[4])) ]
			# Higher boundaries 
			hsvHigh[currentColor] =  [ int(float(fullSeq[1])),int(float(fullSeq[3])),int(float(fullSeq[5])) ]
			file1.close()
			
			# Summary thresholds
			l_b = np.array( hsvLow[currentColor] ) #lower values
			u_b = np.array( hsvHigh[currentColor] ) #upper values
			
			mask = cv2.inRange(hsv, l_b, u_b)
			maskColl[currentColor] = mask
			res = cv2.bitwise_and(im1, im1, mask=mask)
			maskWithColorColl[currentColor] = res
			
			showMasks = 0 # for debugging
			if showMasks: 
				nameWin = 'mask' + currentColor
				cv2.namedWindow(nameWin)
				cv2.imshow(nameWin,mask)
				
			showMasksWithColor = 0 # for debugging
			if showMasksWithColor: 
				nameWin = 'res' + currentColor
				cv2.namedWindow(nameWin)
				cv2.imshow(nameWin,res)
			
		
		if showMasks or showMasksWithColor:
			key = cv2.waitKey(0)
			cv2.destroyAllWindows()
		
		# Postprocessing
		maskCollPost = dict.fromkeys(dictColor)
		for i in range(0, len(dictColor)):
			currentColor = dictColor[i]
			currentMask = maskColl[currentColor]
			kernel = np.ones((5,5),np.uint8)
			maskPost = cv2.morphologyEx(currentMask, cv2.MORPH_OPEN, kernel)
			
			maskCollPost[currentColor] = maskPost
			
			showMasksPost = 0#1
			if showMasksPost:
				nameWin = 'maskPost' + currentColor
				cv2.namedWindow(nameWin)
				cv2.imshow(nameWin,maskPost)
		
		if showMasksPost:
			key = cv2.waitKey(0)
			cv2.destroyAllWindows()
			
		# Extract features and safe in objects
		# Draw contours
		drawing = np.zeros((hsvMain.shape[0], hsvMain.shape[1], 3), dtype=np.uint8)
		# Show in a window
		# cv2.namedWindow('ContoursRaw') # for debugging
		cv2.namedWindow('Detection')
		mainImage4Draw = im1.copy()
		
			
		for i in range(0, len(dictColor)):
			currentColor = dictColor[i]
			currentMaskPost = maskCollPost[currentColor]
			currentMaskWithColor = maskWithColorColl[currentColor]
			
			# Find contours of objects
			cnts = cv2.findContours(currentMaskPost.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)
						
			# Extract features based on contours
			for c in cnts:
				
				# Compute the center of the contour
				M = cv2.moments(c)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				position = [cX, cY]

				x,y,w,h = cv2.boundingRect(c)
				cv2.rectangle(mainImage4Draw,(x,y),(x+w,y+h),(255,0,0),2)
				
				dimension = (w, h)
				aspect_ratio = h/w
				area = cv2.contourArea(c)
				
				# Threshold for filtering very small/ big areas
				if area < 200 or area > 3000: continue
					
				# Loop through all existing objects
				brickAlreadyExists = 0
				for id in dictLegoBricks:
					
					# Check if LegoBrick already exists
					currentBrick = dictLegoBricks[id]
					
					# Intialise bits, 0 means it's not the same brick
					colorBit  = 0
					#areaBit = 0
					#identificationBit = 0
					distanceBit = 0
					
					# Parameter
					#deltaArea = 0.05
					deltaDistance = 25 #pixel
					
					# Check all Bits
					if currentBrick.color == currentColor:
						colorBit = 1
					
					# Allow some noise-related delta
					#if currentBrick.area <= area*(1 + deltaArea) and currentBrick.area >= area*(1 - deltaArea):
					#	areaBit = 1
					
					# Check if the bricks have the same class
					#if currentBrick.identification == identification:
				#		identificationBit = 1
					
					# Use euclidian distance for distance
					current_brick = np.array(currentBrick.position)
					new_brick = np.array(position)
					dist2 = np.linalg.norm(current_brick-new_brick)
					if dist2 <= deltaDistance:
						distanceBit = 1
						
						# TODO: How to handle exact same position
						#if dist2 < 5: 
						#	distanceBit = 2
						#	brickAlreadyExists = 1
						#	break 

					if colorBit and distanceBit:
						brickAlreadyExists = 1
						break
				
				
				if brickAlreadyExists: # Update position
					dictLegoBricks[id].position = position
					currentID = id
					identification = dictLegoBricks[id].identification
					outputString =  'Position of Brick with id: ' + str(currentID) + ' was updated.'
					print('Position of Brick with id was updated.')
				else: # Create a new id
					newIndex = len(dictLegoBricks)
					rgb_image = resize_image_with_padding(currentMaskWithColor[y:y+h, x:x+w])
					legoclass = str(predict_class(rgb_image))
					identification = get_class_from_class_id(int(legoclass))
					print("Creating brick with class ", legoclass)
					tk_image = ImageTk.PhotoImage(image=to_bgr(Image.fromarray(rgb_image).resize((100, 100))))
					currentID = newIndex
					dictLegoBricks[newIndex] = Lego(newIndex, currentColor, position, rgb_image, aspect_ratio, area, legoclass, tk_image)
					outputString = 'Brick does not exist. New ID with value: ' + str(currentID) +' was created. \n' + dictLegoBricks[newIndex].describe()[0]
					print(outputString)

					

				########## Plotting ############
				# Contours
				cv2.drawContours(drawing, [c], -1, (0, 255, 0), 2)
				cv2.drawContours(mainImage4Draw, [c], -1, (0, 255, 0), 2)
				# Circle in the center
				cv2.circle(drawing, (cX, cY), 5, (255, 255, 255), -1)
				cv2.circle(mainImage4Draw, (cX, cY), 7, (255, 255, 255), -1)
				# ID
				y_offset_id = 40
				idString = 'ID: ' + str(round(currentID,0))
				cv2.putText(drawing, idString, (x, y - y_offset_id),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.putText(mainImage4Draw, idString, (x, y - y_offset_id),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				# Color
				y_offset_color = 25
				cv2.putText(drawing, currentColor, (x, y - y_offset_color),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.putText(mainImage4Draw, currentColor, (x, y - y_offset_color),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				# String representation of class
				y_offset_class = 10
				cv2.putText(drawing, identification, (x, y - y_offset_class),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.putText(mainImage4Draw, identification, (x, y - y_offset_class),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				
				alpha = 0.05
				overlay = mainImage4Draw.copy()
				##### corner
				cv2.rectangle(overlay, (0, 0), (0 + 250, 0 + 30), (0,0,0), -1)

				##### putText
				cv2.putText(overlay, f"Pieces in total: {len(dictLegoBricks)}", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

				#### apply the overlay
				cv2.addWeighted(overlay, alpha, mainImage4Draw, 1 - alpha,0, mainImage4Draw)
				
				#cv2.rectangle(mainImage4Draw, (0, 0), (0 + 200, 0 + 30), (0,0,0), -1)
				#cv2.putText(mainImage4Draw, f"Pieces in total: {len(dictLegoBricks)}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				
				
		#cv2.imshow('ContoursRaw', drawing)	# For debugging
		cv2.imshow('Detection', mainImage4Draw)		
		cv2.imshow("Original video",frame)
		
		
		
		##########  Saving images or exit  ##########
		k = cv2.waitKey(1)
		if k%256 == 27:
			# ESC pressed
			break
		elif k %256 == 32:
			# SPACE pressed
			img_name = "image_{}.png".format(img_counter)
			cv2.imwrite(img_name, frame)
			print("{} written!".format(img_name))
			img_counter += 1

cam.release()
cv2.destroyAllWindows()
print(len(dictLegoBricks))

# Header
label = Label(canvas, text=f"{'Id':<15}{'Class':<15}{'Identification':<25}{'Color':<15}{'Position':<15}{'Aspect ratio':<15}{'Area':<15}{'Image':<15}", bg="gray70", font=("bold"), compound=RIGHT)
canvas.create_window(0, y, window=label, anchor=NW)
y += 50

lista = [lego.describe() for lego in dictLegoBricks.values()]

height = len(lista)+1
width = 10

for i in range(height-1): #Rows
	label = Label(canvas,image=lista[i][1], text=lista[i][0], font=(12), compound=RIGHT)
	canvas.create_window(0, y, window=label, anchor=NW)
	y += 100
	


scrollbar = Scrollbar(canvas, orient=VERTICAL, command=canvas.yview)
scrollbar.place(relx=1, rely=0, relheight=1, anchor=NE)
canvas.config(yscrollcommand=scrollbar.set, scrollregion=(0, 0, 0, y))

	
def toplevel():
	top = Toplevel()
	top.title('Click to save image')
	top.wm_geometry("250x500")
	
	ws = root.winfo_screenwidth()
	hs = root.winfo_screenheight()
	w = 800
	h = 500
	x = (ws / 2) - (w / 2)
	y = (hs / 2) - (h / 2)
	#root.geometry('%dx%d+%d+%d' % (w, h, x, y))
	root.update()
	root.configure(background="gray70")
	scframe = VerticalScrolledFrame(top)
	scframe.pack()
	lista = [lego.describe() for lego in dictLegoBricks.values()]
	height = len(lista)+1
	width = 10
	for i in range(height-1): #Rows
		tk_image = lista[i][1]
		lego_class = lista[i][0].split()[1]
		class_desc = get_class_from_class_id(int(lego_class))
		btn = Button(scframe.interior, height=100, width=250, relief=FLAT, bg="gray99", fg="gray70",font="Dosis", text="ID: " + str(i) + f"\t\n{lego_class} ({class_desc})", command=lambda i=i,x=x: openlink(i), image = tk_image, compound=RIGHT)
		btn.pack(padx=10, pady=5, side=TOP)

def openlink(i):
	"""This function is called when a button is clickes and saves a 
	jpg image of the according lego brick.
	"""

	lego = dictLegoBricks[i]
	
	class_id = lego.legoclass
	string_representation = lego.identification

	rgb_image = lego.rgb_image
	dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
	img_name = "{} {} {}.png".format(class_id , string_representation, dt_string)
	cv2.imwrite(f"./own_dataset/{class_id}/" + img_name, rgb_image)
	print("{} written!".format(img_name))
	

toplevel()

root.mainloop()
	
	
