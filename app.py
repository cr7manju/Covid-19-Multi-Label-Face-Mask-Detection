# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:42:40 2020

@author: adhar
"""
# coding=utf-8
from flask import Flask, render_template, Response,request
from werkzeug.utils import secure_filename
from camera_detect import VideoCamera
from image_detect import get_image
import cv2
import os
from image_resize import maintain_aspect_ratio_resize

# Invoke the Flask app
app = Flask(__name__)
UPLOAD_FOLDER = './image_save'
# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    # Main page   
    return render_template('home.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    return render_template('image.html')

def img_gen():
    while True:
        frame=cv2.imread('./static/detectedimgs/detect.jpg')
        # Resizing the image to display with maintaining aspect ratio
        resized = maintain_aspect_ratio_resize(frame, width=800)
        ret, jpeg = cv2.imencode('.jpg', resized)
        frame=jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        

@app.route('/img_feed')
def img_feed():
    return Response(img_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ShowImage',methods= ['GET', 'POST'])
def showimage():
    if request.method == 'POST':
        try:
            for imgs in os.listdir('./image_save/'):
                f='./image_save/'+imgs
                os.remove(f)
        except :
            pass
        image = request.files['myfile']
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        get_image()
        
        return render_template('imageshow.html')
    return render_template('image.html')

@app.route('/video')
def video():
    return render_template('video.html')

def video_gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(video_gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/technology')
def tech():
    return render_template('tech.html')
if __name__ == '__main__':
    app.run(debug=False)
