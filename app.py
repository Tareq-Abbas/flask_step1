from flask import Flask, render_template, request,Response
import cv2
import torch
import numpy as np

import argparse
import io
import os
from PIL import Image
from io import BytesIO


model = torch.hub.load('ultralytics/yolov5', 'custom', 'ultralytics/yolov5/best.pt')
class_names=['D00','D10', 'D20', 'D40']


def detect_damage():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        ret, frame = cap.read()
        result = model(frame)

        # the next line will return xmin,ymin, xmax,ymax,conf,class,name
        # we want it to return the name of the detected damage only
        info = result.pandas().xyxy[0]
        # now we want to return all the detected damages at once so we save them in an array (if detected)
        # we use an if statement because this will give an error if it did not detect any damages
        if len(info["name"]) != 0:
            objects = []
            for name in info["name"]:
                objects.append(name)
            # cls= info["name"].iloc[0]
            cls = objects
            #print(cls)
        # print(info)

        # print(type(result.pandas().xyxy[0]))
        #cv2.imshow('Screen', np.squeeze(result.render()))
        if cv2.waitKey(10) & 0xff == ord('x'):
            break
        if cv2.getWindowProperty('Screen', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


app = Flask(__name__)
cap = cv2.VideoCapture(0)


def generate_frames():
    cap = cv2.VideoCapture(0)
    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        if success == True:

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            # print(results)
            # print(results.pandas().xyxy[0])
            # results.render()  # updates results.imgs with boxes and labels
            #results.print()  # print results to screen
            # results.show()
            # print(results.imgs)
            # print(type(img))
            # print(results)
            # plt.imshow(np.squeeze(results.render()))
            # print(type(img))
            # print(img.mode)

            # convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render())  # RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR

            # print(type(img))
            # print(img.shape)
            # frame = img
            # ret,buffer=cv2.imencode('.jpg',img)
            # frame=buffer.tobytes()
            # print(type(frame))
            # for img in results.imgs:
            # img = Image.fromarray(img)
            # ret,img=cv2.imencode('.jpg',img)
            # img=img.tobytes()

            # encode output image to bytes
            # img = cv2.imencode('.jpg', img)[1].tobytes()
            # print(type(img))
        else:
            break
        # print(cv2.imencode('.jpg', img)[1])

        # print(b)
        # frame = img_byte_arr

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        # print(frame)

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


#this will render the template index.html from the directory templates,
# removing it from that directory will cause error
@app.route('/')
def index():
    return render_template('index.html')


#this will send the frames to the div video in the index.html template
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)