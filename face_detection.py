import cv2

# Load some weights on face frontals from opencv
# Its actually a model
trained_face_data = cv2.CascadeClassifier("dependencies/haarcascade_frontalface_default.xml")

# You can either 
# A) Choose an image to detect faces in
# - img = cv2.imread("dependencies/images/myfamilyphoto.jpg") 

# B) Record from a video
# - the 0 refers to your 'default' or 'first' or 'main' webcam
webcam = cv2.VideoCapture(0)

while True:

	# read the current frame
	# webcam.read() returns two things, 
	# a Boolean of whether the frame was read successfully, and the actual RGB values of the image
	successful_frame_read, frame = webcam.read()

	# Convert your image to grayscale
	grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# The colour of the rectangles that are drawn over the detected objects
	# In OpenCV, colours are in (Blue, Green, Red)
	# rectangle_colour = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

	# Now, to detect faces
	# MultiScale means objects can be detected 'no matter the size of the image'
	# It finds and then returns a numpy array of the detected objects 
	# Which is an array of coordinates of rectangles that enclose each detected object. Observe one example:
	# [[top_left_x
	#	top_left_y 
	#	width_of_rectangle 
	#	height_of_rectangle]] 

	face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

	for image_vector in face_coordinates:
		x, y, w, h = image_vector

		# SYNTAX:
		# cv2.rectangle(img, (top_left_x, top_left_y), (rectangle_width, rectangle_height), (blue, green, red), line_width)
		
		rectangle_colour = (123, 182, 49)
		
		cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_colour, 2)

	# Shows the image in a pop-up window
	cv2.imshow("Face Detector Prototype", frame)

	# cv2.waitKey blocks the execution of your code
	# waitKey can accept a parameter that is the amount of miliseconds it should
	# block the execution of your code before allowing your program to move on to the next part of the code. 
	# You can leave waitKey() empty to make it wait infinitely
	# WaitKey also returns the ASCII value of the key that was pressed 
	key = cv2.waitKey(1)

	# 81 & 113 are the ASCII values of the letter Q
	if key == 81 or key == 113:
		break
			

"""
A haar is a rudimentary building block of an image, it can be an edge feature,
a line feature or a rectangle feature. It is essentially how classification 
algorithms classify images. They break down the image into different haars and 
then cascades from a layer of fine, obscure haars in the neural network to very 
large haars in the final layer to get a 1 or 0 depending on whether or not the 
image is actually a face."""    			