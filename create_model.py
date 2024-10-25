#! C:\Users\Nagendran\Desktop\kidney_Disease_front_end\myenv\Scripts\jupyter.exe
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

from imutils import paths
path_dir=r"C:\Hope_Learning\Deep_learning\Supervised_learning\Face_Mask_Detection"
print("Loading Images From File Directory...")
imgs_path=list(paths.list_images(path_dir))

import pandas as pd
data=[]
Label=[]
for i in imgs_path:
    label=i.split(os.path.sep)[-2] # This line for create target column like with mask and without mask
    Label.append(label)
    image=load_img(i,target_size=(128,128)) # loading the image and changing the image size
    image=img_to_array(image) #this line for convert image into array
    image=preprocess_input(image) # in this line we are using weights(slop and intersept) of mobilenetv2 to our images
    data.append(image)
    
print("With Mask Labels",Label.count("with_mask"))
print("Without Mask Labels",Label.count("without_mask"))

print("Lenth of the data:",len(data))
print("Lenth of the Label:",len(Label))

data=np.array(data,dtype="float32")
Label=np.array(Label)

print("Before convert into number :",Label[0],Label[3828])
lb=LabelBinarizer()
labels=lb.fit_transform(Label)
print("After convert into number :",labels[0],labels[3828])


labels=to_categorical(labels)
print("After convert into to_categorical :",labels[0])
print("After convert into to_categorical :",labels[3828])

print("Train Test Spliting")
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,stratify=labels,random_state=42)


pre_trained_model=MobileNetV2(weights='imagenet',include_top=False,
                             input_tensor=Input(shape=(128,128,3)))

join_our_model=pre_trained_model.output
join_our_model=AveragePooling2D(pool_size=(2,2))(join_our_model)
join_our_model=Flatten(name="flatten")(join_our_model)
join_our_model=Dense(128,activation="relu")(join_our_model)
join_our_model=Dropout(0.5)(join_our_model)
join_our_model=Dense(2,activation="softmax")(join_our_model)

model=Model(inputs=pre_trained_model.input,outputs=join_our_model)

for i in pre_trained_model.layers:
    i.trainable=False
    
    
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
history=model.fit(X_train,y_train,epochs=2,batch_size=100,validation_data=[X_test,y_test])

y_pred=model.predict(X_test)

y_test=np.argmax(y_test,axis=1)

X_test.shape

model.save('mask_detection_model.h5')