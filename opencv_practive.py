import cv2
import numpy as np

specs = cv2.imread('specs.png',-1)
face = cv2.imread('face.png',-1)

gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
eyes_cascade = cv2.CascadeClassifier('frontalEyes.xml')
eyes = eyes_cascade.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in eyes:
    cv2.rectangle(gray, (x,y),(x+w,y+h), (255,86,30),3)

face = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)
specs = cv2.resize(specs, ( w,h))

w,h,c = specs.shape

for i in range(0,w):
    for j in range(0,h):
        if specs[i,j][3] != 0:
            face[y+i, x+j] = specs[i,j]


cv2.imshow('op',face)

cv2.waitKey(0)