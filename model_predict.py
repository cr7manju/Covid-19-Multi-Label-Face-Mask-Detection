# -*- coding: utf-8 -*-
"""
Created on Thu Nov  18 15:57:38 2020

@author: adharsh
"""

import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.applications.vgg19 import preprocess_input 
#from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Function used to preprocess the input image or streaming data and detect faces with model prediction
def prediction(frame,faceNet,model):
    # Get the height and width from the given input frame
    (h, w) = frame.shape[:2]
    # creates 4-dimensional blob from the given input 
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # if confidence value is more than 0.168 preprocess the input image identify the faces and its locations      
        if confidence >= 0.168:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    # Apply the model prediction on detected faces from the input
    for k in faces:
        preds.append(model.predict(k))
    return (locs, preds)
