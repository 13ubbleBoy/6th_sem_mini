import cv2
import pickle
import numpy as np
import os

video=cv2.VideoCapture(0) # capturing video in the variable "video"
facedetect=cv2.CascadeClassifier('/Library/Praveen/VS code/Python/6th_Sem_mini/data/haarcascade_frontalface_default.xml')
faces_data=[] # storing resized images to use further in this list
i=0 # used inside 'if' for sotring resized images

name=input("Enter your name : ")

while True:
    ret,frame=video.read() # reading what is in "video" variable which returns 2 values
    
    frame=cv2.flip(frame, 1) # to fix the left-right

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3, 5) # threshold values
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :] # cropping images (of a single person) so that we can identify them
        resized_img=cv2.resize(crop_img, (50,50)) # resizing the cropped image into '50x50'
        if len(faces_data) <= 100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (240,184,0), 4) # original frame is passed it will show colored image
        # (x,y) are the coordinate values of the image
        # (x+w, y+h) width and height of the channel
        # (50,50,255) color value of 'red'
        # thickness of the border is 1

    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==100:
        break

video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100, -1)

if 'names.pkl' not in os.listdir('/Library/Praveen/VS code/Python/6th_Sem_mini/data/'):
    names=[name]*100
    with open('/Library/Praveen/VS code/Python/6th_Sem_mini/data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('/Library/Praveen/VS code/Python/6th_Sem_mini/data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100 # to collect new data of new user
    with open('/Library/Praveen/VS code/Python/6th_Sem_mini/data/names.pkl', 'wb') as w:
        pickle.dump(names, w)


if 'faces_data.pkl' not in os.listdir('/Library/Praveen/VS code/Python/6th_Sem_mini/data/'):
    with open('/Library/Praveen/VS code/Python/6th_Sem_mini/data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('/Library/Praveen/VS code/Python/6th_Sem_mini/data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('/Library/Praveen/VS code/Python/6th_Sem_mini/data/faces_data.pkl', 'wb') as w:
        pickle.dump(faces, w)
