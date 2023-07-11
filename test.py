from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

video=cv2.VideoCapture(0) # capturing video in the variable "video"
facedetect=cv2.CascadeClassifier('/Library/Praveen/VS code/Python/6th_Sem_mini/data/haarcascade_frontalface_default.xml')


with open('/Library/Praveen/VS code/Python/6th_Sem_mini/data/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('/Library/Praveen/VS code/Python/6th_Sem_mini/data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

while True:
    ret,frame=video.read() # reading what is in "video" variable which returns 2 values
    
    frame=cv2.flip(frame, 1) # to fix the left-right

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3, 5) # threshold values
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)

        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist=os.path.isfile("/Library/Praveen/VS code/Python/6th_Sem_mini/Attendance/Attendance_" + date + ".csv")

        cv2.rectangle(frame, (x,y), (x+w, y+h), (240,184,0), 2)
        cv2.rectangle(frame, (x-1,y-40), (x+w+1, y), (240,184,0), -1)

        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (240,184,0), 2)
        # (x,y) are the coordinate values of the image
        # (x+w, y+h) width and height of the channel
        # (50,50,255) color value of 'red'
        # thickness of the border is 1
        attendance=[str(output[0]), str(timestamp)]
  
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('p'):
        time.sleep(1)
        if exist:
            with open("/Library/Praveen/VS code/Python/6th_Sem_mini/Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("/Library/Praveen/VS code/Python/6th_Sem_mini/Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()

    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
