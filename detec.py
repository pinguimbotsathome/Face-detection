import os
import sys
import cv2
import numpy as np
from time import time
from datetime import datetime as dt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def detec(image):
	prototxt_path = os.path.sep.join([main_dir,'Models/deploy.prototxt'])
	weights_path = os.path.sep.join([main_dir,'Models/res10_300x300_ssd_iter_140000.caffemodel'])
	model = load_model(os.path.sep.join([main_dir,'Models/model_detector.h5']))


	image = cv2.imread(image)
	(h, w) = image.shape[:2]
	

	net = cv2.dnn.readNet(prototxt_path, weights_path)

	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	
	qtd_detec = 0;  labels = []	
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			qtd_detec += 1
		
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

			_, with_mask, without_mask = model.predict(face, batch_size=32)[0] 
		

			if with_mask >= 0.85:
			  color = (0, 128, 0)
			  label = f"{with_mask*100:.1f}%: With mask"
			else:
			  label = f"{(1-with_mask)*100:.1f}%: Without mask"
			  color = (0, 0, 255)
			
			labels.append(label)
			
			aspect_ratio = float(image.shape[1])/float(image.shape[0])
			cv2.putText(image, label, (startX, startY - 10), \
			cv2.FONT_HERSHEY_SIMPLEX, 1.5*aspect_ratio, color, 4)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 10)

	return image, qtd_detec, labels

def loop_folder(folder_in):
	in_dir = f'{main_dir}/{folder_in}'
	out_dir = f'{in_dir}output'
	
	if not os.path.exists(out_dir): 
		os.mkdir(out_dir)
		out_dir = out_dir
	
	
	f = open(f'{in_dir}log.txt', "w")
	f.write(dt.now().strftime('%Y-%m-%d %H:%M:%S\n'))
	
	t1 = time()
	for subdir, dirs, files in os.walk(in_dir):
		if subdir != out_dir:			
			for file in files:
				if file.endswith(".jpg"):
					img, qtd_detec, labels = detec(os.path.join(subdir,file))
					
					os.chdir(out_dir)
					filename = (f'N_{file}')
					cv2.imwrite(filename, img)
					
					f.write(f'\n{filename}: {qtd_detec} detections \n')
					for l in labels: f.write(f'{l}\n')
	t2 = time()           
	print(f'I spent {t2-t1:.2f}s on this joke')

def resize(image, window_height=720):
    aspect_ratio = float(image.shape[1])/float(image.shape[0])
    window_width = window_height/aspect_ratio
    image = cv2.resize(image, (int(window_height),int(window_width)))
    return image
    
    		 		 
def main(pic_folder):
	if pic_folder.endswith("/"):
		loop_folder(pic_folder)	
	
	else:
		image, *_ = detec(pic_folder)	
		cv2.imshow('output',resize(image))
		cv2.waitKey(0);	cv2.destroyAllWindows()	
		 

main_dir = os.getcwd()
main(sys.argv[1])	 
		 
		            
