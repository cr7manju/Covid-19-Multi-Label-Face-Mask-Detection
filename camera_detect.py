# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:57:38 2020

@author: adharsh
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model_predict import prediction
import time
import math

# Class for Streaming Webcam video
class VideoCamera(object):
    
    def __init__(self):
       self.prototxtPath = "./face_detector/deploy.prototxt"
       self.weightsPath ="./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
       self.faceNet = cv2.dnn.readNet(self.prototxtPath,self.weightsPath)
       self.model=load_model('mv2_mask_detection.hdf5')  
       #self.model=load_model('vgg19_mask_detection.hdf5') 
       #self.model=load_model('effb7_mask_detection.hdf5') 
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        cv2.destroyAllWindows()
        self.video.release()  
        
    # Get the webcamera video output apply model prediction on the frame and display with labels
    def get_frame(self):
        try:
            ret, frame = self.video.read()
            now=time.time()
            (locs, preds) = prediction(frame,self.faceNet,self.model)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                cla=np.argmax(pred[0])
                label = "No Mask- No Entry" if cla==0 else "Wrong Mask- Not Safe" if cla==1 else "Mask- Safe"
                color = (0, 0, 255) if cla == 0 else (0, 255, 255) if cla==1 else (255, 255, 0)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(pred[0]) * 100)

                # Create Text around the bounding box displaying label with probability percentage 
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.line(frame,(startX,startY),(startX,startY+25),color,2)
                cv2.line(frame,(startX,startY),(startX+25,startY),color,2)
        
                cv2.line(frame,(endX,startY),(endX,startY+25),color,2)
                cv2.line(frame,(endX,startY),(endX-25,startY),color,2)
        
                cv2.line(frame,(startX,endY),(startX,endY-25),color,2)
                cv2.line(frame,(startX,endY),(startX+25,endY),color,2)
        
                cv2.line(frame,(endX, endY),(endX,endY-25),color,2)
                cv2.line(frame,(endX, endY),(endX-25,endY),color,2)
        
        
             #  cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            (hei, wid) = frame.shape[:2]
            #fps=cap.get(cv2.CAP_PROP_FPS)
            end=time.time()
            f=1/(end-now)
            FPS='FPS : '+str(math.ceil(f))
            # Display the Frames Per Second below the streaming video
            cv2.putText(frame,str(FPS),(0,hei-20),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 1)
            no_faces='No. of faces in video   : '+str(len(locs))
            # Display the Number of people on the webcam of streaming video
            cv2.putText(frame,str(no_faces),(80,hei-20),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 1)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        except :
            pass