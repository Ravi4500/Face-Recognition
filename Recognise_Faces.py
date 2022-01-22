import cv2
import numpy as np
import face_recognition
import os

path = 'Trained_images'
images = []
Names_list = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    Names_list.append(os.path.splitext(cl)[0])


def findEncodings(images):
    list_of_encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        list_of_encodings.append(encode)
    return list_of_encodings


Trained_encodings = findEncodings(images)
print("Encodings of images to be trained are finished")

cam = cv2.VideoCapture(0)
print(cam.isOpened())
while cam.isOpened():
    success, img = cam.read()
    Small_image = cv2.resize(img,(0,0),None,0.5,0.5)
    Small_image = cv2.cvtColor(Small_image, cv2.COLOR_BGR2RGB)
    face_loc = face_recognition.face_locations(Small_image)
    encodings = face_recognition.face_encodings(Small_image,face_loc)

    for encodes, faceLoc in zip(encodings,face_loc):
        matches = face_recognition.compare_faces(Trained_encodings,encodes)
        dist = face_recognition.face_distance(Trained_encodings,encodes)
        matched_index = np.argmin(dist)

        if matches[matched_index]:
            name = Names_list[matched_index].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc

            y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 255, 0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)


