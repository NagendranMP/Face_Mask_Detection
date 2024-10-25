#! C:\Users\Nagendran\Desktop\kidney_Disease_front_end\myenv\Scripts\jupyter.exe
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from imutils.video import VideoStream
import argparse
import imutils
import time
import os

def detect(frame,faceNet,maskNet):
    h,w=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),104.0,177.0,123.0)

    faceNet.setInput(blob)
    detections=faceNet.forward()

    faces=[]
    locs=[]
    preds=[]

    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
    if len(faces)>0:
        faces=np.array(faces,dtype="float32")
        preds=maskNet.predict(faces,batch_size=32)

    return locs,preds
ap=argparse.ArgumentParser()
ap.add_argument("-f","--face",type=str,default="face_detector",help="path to face detector model directory")
ap.add_argument("-m","--model",type=str,default="mask_detection_model",help="path to trained face mask detector model")
ap.add_argument("-c","--confidence",type=float,default="0.5",help="minimum probability to filter weak detection")
arg=vars(ap.parse_args())

print("[INFO] loading face ditector model...")
prototxtPath=os.path.sep.join([arg["face"],"deploy.prototxt"])
weightPath=os.path.sep.join([arg["face"],"res10_300x300_ssd_iter_140000.caffemodel"])

faceNet = cv2.dnn.readNet(prototxtPath, weightPath)


print("[INFO] loading face mask detector model...")
maskNet=load_model(arg["mask_detection_model"])

print("[INFO] starting vedio streem...")
vs=VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame=vs.read()
    frame=imutils.resize(frame,width=400)
    locs,preds=detect(frame,faceNet,maskNet)

    for (box,pred) in zip(locs,preds):
        startx,starty,endx,endy=box
        mask,withoutmask=pred

        label="Mask" if mask>withoutmask else "No Mask"
        color=(0,255,0) if label== "Mask" else (0,0,255)

        if(label=="Mask"):
            cv2.putText(frame,"Mask: You Are Allowed",(startx,starty-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
            cv2.rectangle(frame,(startx,starty),(endx,endy),color,2)
        elif(label=="No Mask"):
            cv2.putText(frame,"No Mask: You Are Not Allowed",(startx,starty-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
            cv2.rectangle(frame,(startx,starty),(endx,endy),color,2)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()

vs.release()