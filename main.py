#from importlib.resources import path
import torch
#from matplotlib import pyplot as plt
import numpy as np
import cv2

#as we can see from the path to yolov5, yolov5 exists in a folder called ultralytics
#ultralytics contains only yolov5
# but if we but yolov5 in the project folder (removerd ultralytics and replaced it with its content yolov5)
# we will get error: expect2 get 1
#so we have to put yolov5 inside ultralytics
model = torch.hub.load('ultralytics/yolov5', 'custom', 'ultralytics/yolov5/best.pt')
#model = detect.load('ultralytics/yolov5/best.pt')


class_names=['D00','D10', 'D20', 'D40']


cap= cv2.VideoCapture(0)
while cap.isOpened():

    ret, frame = cap.read()
    result= model(frame)

    #the next line will return xmin,ymin, xmax,ymax,conf,class,name
    #we want it to return the name of the detected damage only
    info = result.pandas().xyxy[0]
    #now we want to return all the detected damages at once so we save them in an array (if detected)
    # we use an if statement because this will give an error if it did not detect any damages
    if len(info["name"]) != 0:
        objects = []
        for name in info["name"]:
            objects.append(name)
        #cls= info["name"].iloc[0]
        cls= objects
        print(cls)
    #print(info)

    #print(type(result.pandas().xyxy[0]))
    cv2.imshow('Screen',np.squeeze(result.render()))
    if cv2.waitKey(10) & 0xff ==ord('x'):
        break
    if cv2.getWindowProperty('Screen', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()