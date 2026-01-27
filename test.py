import cv2 as cv
import numpy as np

image = cv.imread("damage_car_pic1.jpg")
imager = cv.resize(image,(640,320))
cv.imshow("display_someCar",imager)
k = cv.waitKey(0)

cap = cv.VideoCapture('sample2.mp4')
 
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.resize(frame,(640,320))
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
 
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()