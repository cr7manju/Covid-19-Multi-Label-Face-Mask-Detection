# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:57:38 2020

@author: adharsh
"""

import cv2
from tensorflow.keras.models import load_model
from model_predict import prediction
import os
import numpy as np
prototxtPath = "./face_detector/deploy.prototxt"
weightsPath ="./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath,weightsPath)
model=load_model('mv2_mask_detection.hdf5')
#model=load_model('vgg19_mask_detection.hdf5')
#model=load_model('effb7_mask_detection.hdf5')
# Function to detect the person in image wearing mask or not
def get_image():
    f=[]
    for file in os.listdir('./image_save/'):
        f.append(file)
    path="./image_save/"+f[0]
    frame=cv2.imread(path,cv2.IMREAD_COLOR)
    try:
        (locs, preds)=prediction(frame,faceNet,model)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            cla=np.argmax(pred[0])
            label = "No Mask" if cla==0 else "Wrong Mask" if cla==1 else "Mask"
            color = (0, 0, 255) if cla == 0 else (0, 255, 255) if cla==1 else (255, 255, 0)


    		# include the probability in the label
            label = "{}: {:.2f}%".format(label, max(pred[0]) * 100)
            cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        try:
            for i in os.listdir('./static/detectedimgs/'):
                file='./image_save/detectedimgs/'+i
                os.remove(file)          
        except:
            pass
        # Write the output labelled image to detected images directory in static folder
        cv2.imwrite('./static/detectedimgs/detect.jpg',frame)
    except :
        pass