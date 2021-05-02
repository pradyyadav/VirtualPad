import cv2
import numpy as np
import os
import time
import handTrackingModule as htm

###############
brushThickness = 15
eraserThickness = 75
########3


folderPath = "header"

myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    img = cv2.imread(f'{folderPath}/{imPath}')
    img = cv2.resize(img,(640,125))
    overlayList.append(img)

header = overlayList[0]
drawColor = (255,0,255)

xp,yp = 0,0

imgCanvas = np.zeros((480,640,3),np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(3,480)

detector = htm.handDetector(detectionCon=0.85)

while True:

    # 1. Import image
    success, img =  cap.read()
    img = cv2.flip(img,1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)


    # 4. If Selection mode : Two fingers are up
        if fingers[1] and fingers[2]:
            print("Selection mode")
            if y1 < 125:
                if x1 > 120 and x1 < 220:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif x1 > 250 and x1 < 350:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif x1 > 370 and x1 < 450:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif x1 > 520 and x1 < 600:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img,(x1,y1-10),(x2,y2+10),drawColor,cv2.FILLED)
            xp,yp = x1,y1
            
    # 5. If Drawing mode : Index finger is up
        elif fingers[1]:
            print("Drawing mode")
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            if xp == 0 and yp == 0:
                xp,yp = x1,y1
            if drawColor == (0,0,0):
                imgCanvas = cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
                img = cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                imgCanvas = cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
                img = cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgThresh = cv2.threshold(imgGray,10,255,cv2.THRESH_BINARY_INV)
    imgThresh = cv2.cvtColor(imgThresh,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgThresh)
    img = cv2.bitwise_or(img,imgCanvas)
    #img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

    ## Setting header image
    img[0:125,0:640] = header
    cv2.imshow('virtualPainter', img)
    cv2.waitKey(1)
