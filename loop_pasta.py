import cv2
import sys
import time
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def main(image):
	prototxt_path = "/home/doc/Documents/GitHub/Face-detection/deploy.prototxt"
	weights_path = "/home/doc/Documents/GitHub/Face-detection/res10_300x300_ssd_iter_140000.caffemodel"

	net = cv2.dnn.readNet(prototxt_path, weights_path)


	model = load_model('/home/doc/Documents/GitHub/Face-detection/model_detector.h5')


	image = cv2.imread(image)#(sys.argv[1])
	(h, w) = image.shape[:2]


	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))


	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			mask_weared_incorrect, with_mask, without_mask = model.predict(face, batch_size=32)[0] 
	    
	    # extract the face ROI, convert it from BGR to RGB channel
			if max([mask_weared_incorrect, with_mask, without_mask]) == with_mask:
			  label = 'with_mask'
			  color = (0, 255, 0)
			elif max([mask_weared_incorrect, with_mask, without_mask]) == without_mask:
			  label = 'without_mask'
			  color = (0, 0, 255)
			else:
			  label = 'mask_worn_incorrectly'
			  color = (255, 140, 0)
			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max([mask_weared_incorrect, with_mask, without_mask]) * 100)
			# display the label and bounding box rectangle on the output frame
			cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 2, color, 10)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 5)


	def resize(image, window_height=1280):
	    aspect_ratio = float(image.shape[1])/float(image.shape[0])
	    window_width = window_height/aspect_ratio
	    image = cv2.resize(image, (int(window_height),int(window_width)))
	    return image
	    
	image = resize(image)  
	return image  

import os
directory = '/home/doc/Documents/GitHub/Face-detection/dataset/Raiz'
N_directory = '/home/doc/Documents/GitHub/Face-detection/dataset/Nut'

for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".jpg"):
            #print(subdir)
            
            #image_path = os.path.join(subdir,file)
            #image = cv2.imread(image_path)
            #cv2.imshow('output',image)
            #cv2.waitKey(0); cv2.destroyAllWindows()
            
            #img = cv2.imread(image_path)
            img = main(os.path.join(subdir,file))
            
            os.chdir(N_directory)
            filename = (f'Nut_{file}')
            cv2.imwrite(filename, img)
            
            

	





















