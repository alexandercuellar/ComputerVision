import cv2
import numpy as np

capture = cv2.VideoCapture(1)
capture2 = cv2.VideoCapture(2)




while True:

    ret, image = capture.read()
    ret, image2 = capture2.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.medianBlur(gray,5)
    grial = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)


    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,10)
   
    if circles is not None: 
        for i in circles[0,:]:
            center = (i[0], i[1])
            radius = int(i[2])
            center = tuple(map(int, center)) # asegurarse de que center sea una tupla de enteros
            cv2.circle(image, center, 1,(255,0,0),3)
            cv2.circle(image, center, radius,(255,0,0),3)
            cv2.circle(image2, center, 1,(255,0,0),3)
            cv2.circle(image2, center, radius,(255,0,0),3)
            print (i[0], i[1])

    cv2.imshow('video', image)
    cv2.imshow('video', image2)
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
       
capture.release()  
cv2.destroyAllWindows()