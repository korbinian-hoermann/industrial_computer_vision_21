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


##############################  Variables  #############################

# Dictionary of classes
classes_dict = {
    0: "plate 2x4",
    1: "roof tile 2x2",
    2: "plate 1x2",
    3: "tire 1x1",
    4: "U 2x4", 
    5: "A 2x3"
}


##############################  Classes  ###############################
class Lego: 
  """This class will be used as our data structure to safe information about detected lego."""
  def __init__(self, id, color, position, rgb_image, aspect_ratio, area, tk_image):
	  self.id = id #int
	  self.color = color #string
	  self.position = position #[x,y] of middlepoint (from findcontours)
	  self.rgb_image = rgb_image #3dim array/matrix
	  self.tk_image = tk_image #3dim array/matrix
	  self.aspect_ratio = round(aspect_ratio,2) #float
	  self.area = area #int - number of pixels
	  
  def describe(self):
	  """Returns a tuple describing the lego piece 
	  (<info as string>, tk_image)
	  """
	  
	  return (f"{self.id:<15}{self.color:<20}{str(self.position):<20}{self.aspect_ratio:<15}{self.area:<10}\t", self.tk_image)
	  
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
	

############################  Camera setup  ############################
cam = cv2.VideoCapture(0)
cv2.namedWindow("Original video")
#cv2.namedWindow("Detection video")

img_counter = 0
dictLegoBricks = {}

print("=============================")
print("Defined classes:")
print("-----------------------------")
for id, lego in classes_dict.items(): 
	print(f"{id}:\t{lego}")
print("=============================\n")
	

class_id = int(input("For which Legopiece images should be created (insert id): "))
string_representation = classes_dict[class_id]


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




time.sleep(1)
ret, frame = cam.read()
time.sleep(1)

while True:
	# Get frame from camera
	ret, frame = cam.read()
	if not ret: 
		print("failed to grab frame")
		break
		
	# Preprocess image
	im1 = ResizeWithAspectRatio(frame, width=500)
	im2 = cv2.bilateralFilter(im1,25,75,75) 
	preprocessed_frame = cv2.medianBlur(im2, 5)
	
	# Create binary masks via HSV
	hsvMain = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2HSV)
	dictColor = ['yellow', 'orange', 'red', 'blue', 'green']

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
			if area < 200 or area > 3500: continue
				
			# Loop through all existing objects
			brickAlreadyExists = 0
			for id in dictLegoBricks:
				
				# Check if LegoBrick already exists
				currentBrick = dictLegoBricks[id]
				
				# Intialise bits, 0 means it's not the same brick
				colorBit  = 0
				areaBit = 0
				identificationBit = 0
				distanceBit = 0
				
				# Parameter
				deltaArea = 0.05
				deltaDistance = 1 #pixel
				
				# Check all Bits
				if currentBrick.color == currentColor:
					colorBit = 1
				
				# Allow some noise-related delta
				if currentBrick.area <= area*(1 + deltaArea) and currentBrick.area >= area*(1 - deltaArea):
					areaBit = 1
				
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
					if dist2 < 5: 
						distanceBit = 2
						brickAlreadyExists = 1
						break 

				if colorBit and areaBit and distanceBit:
					brickAlreadyExists = 1
					break
			
			
			if brickAlreadyExists: # Update position
				dictLegoBricks[id].position = position
				currentID = id
				outputString =  'Position of Brick with id: ' + str(currentID) + ' was updated.'
				print('Position of Brick with id was updated.')
			else: # Create a new id
				newIndex = len(dictLegoBricks)
				rgb_image = resize_image_with_padding(currentMaskWithColor[y:y+h, x:x+w])
				tk_image = ImageTk.PhotoImage(image=to_bgr(Image.fromarray(rgb_image).resize((100, 100))))
				currentID = newIndex
				dictLegoBricks[newIndex] = Lego(newIndex, currentColor, position, rgb_image, aspect_ratio, area, tk_image)
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
			y_offset_id = 30
			idString = 'ID: ' + str(round(currentID,0))
			cv2.putText(drawing, idString, (x, y - y_offset_id),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.putText(mainImage4Draw, idString, (x, y - y_offset_id),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			# Color
			y_offset_color = 15
			cv2.putText(drawing, currentColor, (x, y - y_offset_color),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(mainImage4Draw, currentColor, (x, y - y_offset_color),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			# String representation of class
			#y_offset_class = 10
			#cv2.putText(drawing, identification, (x, y - y_offset_class),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			#cv2.putText(mainImage4Draw, identification, (x, y - y_offset_class),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			
	#cv2.imshow('ContoursRaw', drawing)	# For debugging
	cv2.imshow('Detection', mainImage4Draw)		
	cv2.imshow("Original video",frame)
	
	
	
	##########  Saving image or exit  ##########
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

##########  Tkinter GUI Overview  ##########
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
		btn = Button(scframe.interior, height=100, width=200, relief=FLAT, bg="gray99", fg="gray70",font="Dosis", text="ID: " + str(i) + "\t", command=lambda i=i,x=x: openlink(i), image = tk_image, compound=RIGHT)
		btn.pack(padx=10, pady=5, side=TOP)

def openlink(i):

	lego = dictLegoBricks[i]
	print(lego.describe()[0])
	rgb_image = lego.rgb_image
	dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
	img_name = "{} {} {}.png".format(class_id , string_representation, dt_string)
	cv2.imwrite("./own_dataset/{}/".format(class_id) + img_name, rgb_image)
	print("{} written!".format(img_name))
	

toplevel()

root.mainloop()
	
	

