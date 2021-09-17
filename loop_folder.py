import os
import cv2
import numpy as np
from time import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def main(image):
	prototxt_path = os.path.sep.join([main_dir,'Models/deploy.prototxt'])
	weights_path = os.path.sep.join([main_dir,'Models/res10_300x300_ssd_iter_140000.caffemodel'])
	model = load_model(os.path.sep.join([main_dir,'Models/model_detector.h5']))


	image = cv2.imread(image)
	(h, w) = image.shape[:2]
	

	net = cv2.dnn.readNet(prototxt_path, weights_path)

	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()

	for i in range(detections.shape[2]):
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

			_, with_mask, without_mask = model.predict(face, batch_size=32)[0] 
		

			if with_mask >= 0.85:
			  color = (0, 128, 0)
			  label = f"{with_mask*100:.1f}%: With mask"
			else:
			  label = f"{(1-with_mask)*100:.1f}%: Without mask"
			  color = (0, 0, 255)
			
			aspect_ratio = float(image.shape[1])/float(image.shape[0])
			cv2.putText(image, label, (startX, startY - 10), \
			cv2.FONT_HERSHEY_SIMPLEX, 1.5*aspect_ratio, color, 4)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 5)
    
	return image  

main_dir = os.getcwd()
directory = os.path.sep.join([main_dir,'/dataset/Raiz'])
N_directory = os.path.sep.join([main_dir,'/dataset/Nut'])

t1 = time()
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".jpg"):
            img = main(os.path.join(subdir,file))
            
            os.chdir(N_directory)
            filename = (f'Nut_{file}')
            cv2.imwrite(filename, img)
t2 = time()           
print(f'I spent {t2-t1:.2f}s on this joke')
           
                       
