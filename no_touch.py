



#https://github.com/zyrbreyes/yolov5fish/blob/main/app.py
#https://www.youtube.com/watch?v=xzN_aG917-8

import cv2
import torch
import numpy as np



model = torch.hub.load('ultralytics/yolov5', 'custom', 'ultralytics/yolov5/best.pt')
class_names=['D00','D10', 'D20', 'D40']


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
        print(cls)
    # print(info)

    print(type(result))
    cv2.imshow('Screen', np.squeeze(result.render()))
    if cv2.waitKey(10) & 0xff == ord('x'):
        break
    if cv2.getWindowProperty('Screen', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
