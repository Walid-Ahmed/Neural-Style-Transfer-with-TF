# USAGE
# python neural_style_transfer_video_from_file.py --models models  --everyNFrame 10  --videoFileName Shikabala.mp4 --outputFile demo.mp4
#src https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/


# import the necessary packages

import itertools
import argparse
import cv2
import numpy as np
import os



fourcc = cv2.VideoWriter_fourcc(*'XVID')

width=640
height=360  
plusFrameWidth=80
# frame showing input output and the original painting
fullFrame=np.zeros([width*2+plusFrameWidth,height*2,3],dtype=np.uint8)
fullFrame.fill(255)
# frame where we we will draw the plus sign
plusFrame=np.zeros([height,plusFrameWidth,3],dtype=np.uint8)
plusFrame.fill(255)
#frame where we will display outout
bottomFrame = np.zeros([height+20,width*2+plusFrameWidth,3],dtype=np.uint8)
bottomFrame.fill(255)



 


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--modelsFolder", default="models",
	help="path to folder  of  neural style transfer models")
ap.add_argument("--everyNFrame", type=int, default=1,
	help="Sampling rate ")
ap.add_argument("--videoFileName", type=str, default="Shikabala.mp4",
	help="Input Video file ")
ap.add_argument("--outputFile", type=str, default="demo.mp4",
	help="Output Video file ")
args = vars(ap.parse_args())

everyNFrame=args["everyNFrame"]
videoFileName=args["videoFileName"]
modelsFolder=args["modelsFolder"]
outputFile=args["outputFile"]


video_creator = cv2.VideoWriter(outputFile,fourcc, 30, (width*2+plusFrameWidth,height*2+20))


# grab the paths to all neural style transfer models in our 'models'
# directory, provided all models end with the '.t7' file extension
modelPaths=[]
for file in os.listdir(modelsFolder):
    if file.endswith(".t7"):
        modelPaths.append(os.path.join(modelsFolder,file))



# generate unique IDs for each of the model paths, then combine the
# two lists together
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

# use the cycle function of itertools that can loop over all model
# paths, and then when the end is reached, restart again
modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)
fileName=os.path.basename(modelPath)
filename, file_extension = os.path.splitext(fileName)
imgFile=filename+".jpeg"
# Using cv2.imread() method 
originalFrame = cv2.imread(os.path.join("models",imgFile)) 
#print(originalFrame.shape)

# Displaying the image 

# load the neural style transfer model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(modelPath)

# initialize the video stream, then allow the camera sensor to warm up
#fileName="Shikabala.mp4"
print("[INFO] starting video stream...")

print("[INFO] {}. {}".format(modelID + 1, modelPath))

cap = cv2.VideoCapture(videoFileName)

fps = cap.get(cv2.CAP_PROP_FPS)
print("[INFO] Frames per second {} ".format(fps))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
modifidedDuration=frame_count/everyNFrame
modelChangeFreq=(modifidedDuration//9)*fps

print("[INFO] Total number of frames {} ".format(frame_count))
print("[INFO] Duration displayed  in seconds {} ".format(modifidedDuration))
print("[INFO] Frames will be sampled every   {}  frames ".format(everyNFrame))





if (cap.isOpened()== False): 
  print("Error opening video stream or file")

 
# Read until video is completed
frameNum=0
while(cap.isOpened()):

  # Capture frame-by-frame
	ret, frame = cap.read()
	frameNum=frameNum+1

	if(not (frameNum%everyNFrame)==0):
		continue




	# resize the frame to have a width of 600 pixels 


	if(frame is None):
		break
	height, width, layers = frame.shape
	frame = cv2.resize(frame, (600, height))	
	(h, w) = frame.shape[:2]

	# construct a blob from the frame, set the input, and then perform a
	# forward pass of the network
	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
		(103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()

	# reshape the output tensor, add back in the mean subtraction, and
	# then swap the channel ordering
	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
	output = output.transpose(1, 2, 0)


	fullFrame=cv2.resize(output,(width,height))
	fullFrame=fullFrame.astype( np.uint8)

	frame=cv2.resize(frame,(width,height))
	output=cv2.resize(output,(width,height))
	originalFrame=cv2.resize(originalFrame,(width,height))


	output_n = cv2.normalize(src=output, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	demoFrame=np.hstack((frame,plusFrame,originalFrame))
	demoFrame=demoFrame.astype( np.uint8)



	space=300
	bottomFrame[10:10+360,space:space+640,:]=output_n[:,:,:]



	fullFrame=np.vstack((demoFrame,bottomFrame))
	cv2.putText(fullFrame, '+', (644,200), cv2.FONT_HERSHEY_SIMPLEX ,  3, (255, 0, 0) , 2, cv2.LINE_AA)
	cv2.putText(fullFrame, '=', (75,600), cv2.FONT_HERSHEY_SIMPLEX ,  3, (255, 0, 0) , 2, cv2.LINE_AA)
	cv2.imshow('Demo Neural Style Transfer', fullFrame)


	video_creator.write(fullFrame)


	

	#Automaticall change model
	if (frameNum%(modelChangeFreq)==0):
		# grab the next nueral style transfer model model and load it
		(modelID, modelPath) = next(modelIter)
		fileName=os.path.basename(modelPath)
		print("[INFO] {}. {}".format(modelID + 1, modelPath))
		print(fileName)
		filename, file_extension = os.path.splitext(fileName)
		filename, file_extension = os.path.splitext(fileName)
		imgFile=filename+".jpeg"
		originalFrame = cv2.imread(os.path.join("models",imgFile)) 
		net = cv2.dnn.readNetFromTorch(modelPath)

	# if the `q` key was pressed, break from the loop
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
    
# cleanup
cv2.destroyAllWindows()
video_creator.release()
print("[INFO] Output demo saved to  file  {}   ".format(outputFile))
