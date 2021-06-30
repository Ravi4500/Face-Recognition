import cv2
import numpy as np
import face_recognition
import os

path = 'Advancedimages'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print("Encoding finished")

cam = cv2.VideoCapture(0)
print(cam.isOpened())
while cam.isOpened():
    success, img = cam.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    face_loc2 = face_recognition.face_locations(imgS)
    #print(face_loc2)
    encode1 = face_recognition.face_encodings(imgS,face_loc2)
    #print(encode1)

    for encodes, faceLoc in zip(encode1,face_loc2):
        matches = face_recognition.compare_faces(encodeListKnown,encodes)
        print(matches)
        dist = face_recognition.face_distance(encodeListKnown,encodes)
        #print(dist)
        matchindex = np.argmin(dist)

        if matches[matchindex]:
            name = classNames[matchindex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc

            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

   # print(face_loc2)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)


