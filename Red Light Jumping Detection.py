
# import the necessary packages
from tracking_scripts.centroidtracker import CentroidTracker
from tracking_scripts.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import imutils
import dlib
import cv2
import os
import classifier


input="test.mp4"

base="yolo-coco"
labelsPath = os.path.sep.join([base, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([base, "yolov3.weights"])
configPath = os.path.sep.join([base, "yolov3.cfg"])

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] #retrieve yolo layers

print("[INFO] opening video file...")
vs = cv2.VideoCapture(input)

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1]

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if input is not None and frame is None:
		break
	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	traffic_light=0
	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	status = "Waiting"
	rects = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames %10 == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame,1 / 255.0, (416, 416),swapRB=True, crop=False)
		net.setInput(blob)
		layeroutputs = net.forward(ln)
		confidences=[]
		boxes=[]
		classID=[]
		classes=["car","traffic light","bus","truck"]
		# loop over the detections
		for layer in layeroutputs:
			for i, detection in enumerate(layer):

				class_scores=detection[5:]
				confidence = detection[4]
				class_id=np.argmax(class_scores)
				class_score=class_scores[class_id]
				if LABELS[class_id] not in classes:
					continue
				if (confidence*class_score)>0.5:

					confidences.append(float(confidence))
					BOX=detection[0:4]*np.array([W,H,W,H])
					(centerX,centerY,Width,Height)=BOX.astype("int")

					startX=int(centerX-(Width/2))
					startY=int(centerY-(Height/2))
					boxes.append([startX,startY,int(Width),int(Height)])
					classID.append(class_id)

		idxs=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				endX =  x + w
				endY = y + h
				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(x, y, endX, endY)
				tracker.start_track(rgb, rect)


				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append((tracker,LABELS[classID[i]]))

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker,id in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"
			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()
			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			rects.append((startX, startY, endX, endY,id))

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they jumped the red light or not
	cv2.line(frame, (20, int(H //3*1.1)), (350, int(H // 3)), (0, 50, 255), 1)
	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids

	objects= ct.update(rects)
	labels = ct.labels
	boundingboxes=ct.boundingbox
	light_color = ""
	# loop over the tracked objects

	for (objectID, centroid) in objects.items():
		box = boundingboxes.get(objectID)
		#skip traffic lights we don't want to track
		if labels.get(objectID)=="traffic light" and objectID !=2:
			continue
		#draw bounding box for each object
		cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=1)

		if labels.get(objectID)=="traffic light" and objectID ==2:
			traffic_light = frame[box[1]:box[3],box[0]:box[2]]
			color = classifier.get_misclassified_images(traffic_light)
			if color=="red":
				light_color="Red"
			if color == "yellow":
				light_color = "Yellow"
			if color == "green":
				light_color = "Green"
			cv2.putText(frame, light_color, (centroid[0] +5, centroid[1] - 20),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			#when the light turns red, save the first position for each vehicle
			if light_color=="Red" and to.firstpos==0:
				to.firstpos=centroid[1]
			to.centroids.append(centroid)

			if to.counted==True :  #when the vehicle passed the line, we mark it with red color bounding box.
				cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)

			# check to see if the object has been counted or not and the first position is below the line
			if not to.counted and to.firstpos >int(H //3)  : #check the start point
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line,and the light color is red count the object

				if direction < -3 and centroid[1] < int(H //3) and light_color=="Red" :
					totalUp += 1
					to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Vuot Den Do", totalUp),
	    ]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Traffic",frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# close any open windows
cv2.destroyAllWindows()