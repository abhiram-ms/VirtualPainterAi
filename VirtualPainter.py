import cv2
import time
import numpy as np
import os
import mediapipe as mp
import HandTrackingModule as hm
import math
# import cvzone
# from cvzone.SelfiSegmentationModule import SelfiSegmentation

#######################################
brushThickness  = 15
eraserThickness = 50
drawColor = (0,0,255)

####################################
xp,yp = 0,0

Ptime = 0 #PREVIOUS TIME
Ctime =0 #CURRENT TIME

overLayList = []
overLayList_side = []

#image folder path 
folderPath = "assets"
folderPath_side = "assets_side"
myList = os.listdir(folderPath)
myList_side = os.listdir(folderPath_side)

#import images
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overLayList.append(image)

#import images
for imPaths in myList_side:
    image_side = cv2.imread(f"{folderPath_side}/{imPaths}")
    overLayList_side.append(image_side)

#initial 
header = overLayList[0]
sidebar = overLayList_side[0]
sidebar_id = 0
toggle_bars = 0
detector = hm.handDetector(detectionConfidence=0.85)
cap = cv2.VideoCapture(0)

imgCanvas = np.zeros((720,1280,3),np.uint8)
# bg_img = cv2.imread("whitejpg.jpg")
# seg = SelfiSegmentation()
    
while True:
    #1 = import image 
    success,im = cap.read()
    im = cv2.resize(im,(1280,720))
    img = cv2.flip(im,1)
    # img = seg.removeBG(imp, bg_img,threshold=0.8)
    
    #2 = Trcak hands
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    
    if len(lmList) !=0:
        #print(lmList)
        #find tip of index finger and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        x0,y0 = lmList[4][1:]
        
        #3 = check if fingers up
        fingers = detector.fingersUp()
        # if len(fingers) != 0:
        #     print(fingers)
            
        #4 = selection mode - two finger
        if(fingers[1] and fingers [2]):
            xp,yp = 0,0
            
            if y1 < 125:
                toggle_bars = 0
                if 160<x1<380: #red brush selected 
                    header = overLayList[0]
                    drawColor = (0,0,255)
                    
                elif 460<x1<700: # blue brush selected
                    header = overLayList[1]
                    drawColor = (255,0,00)
                
                elif 790<x1<1025: #green brush selected
                    header = overLayList[2]
                    drawColor = (0,255,0)
                
                elif 1090<x1<1260: #white eraser selected
                    header = overLayList[3]
                    drawColor = (0,0,0)
            
            if x1 > 1216:
                toggle_bars = 1
                if 130<y1<255: # square
                    sidebar = overLayList_side[0]
                    sidebar_id = 0
                    
                
                elif 300<y1<425: # circle
                    sidebar = overLayList_side[1]
                    sidebar_id = 1
                    
                
                elif 470<y1<595: # triangle
                    sidebar = overLayList_side[2]
                    sidebar_id = 2
                    
                    
            cv2.rectangle(img, (x1,y1-25),(x2,y2+25), drawColor, cv2.FILLED)
            
        #5 = drawing mode - one finger 
        if fingers[1] and fingers[2] == False:
            cv2.circle(img,(x1,y1), 15,drawColor,cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp,yp = x1,y1
                
            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:   
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
                      
            xp,yp = x1,y1
            
        #6 draw shapes
        if fingers[1] and fingers[3]:
            if sidebar_id == 0:
                cv2.rectangle(img, (x1,y1),(x2,y2), drawColor, cv2.FILLED)
                cv2.rectangle(imgCanvas, (x1,y1),(x2,y2), drawColor, cv2.FILLED)
            elif sidebar_id == 1:
                p1 = [x1,y1]
                p2 = [x0,y0]
                distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
                cv2.circle(img,(x1,y1),distance,drawColor,cv2.FILLED)
                cv2.circle(imgCanvas,(x1,y1),distance,drawColor,cv2.FILLED)
            elif sidebar_id == 2:
                cv2.line(img,(x0,y0),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(x0,y0),(x1,y1),drawColor,brushThickness)
                
            
    imgGrey = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)  
    _, imgInv = cv2.threshold(imgGrey,50,255,cv2.THRESH_BINARY_INV)  
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR) 
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)   
    
            
    #showing panel on top
    img[0:125 ,0:1280] = header
    #showing panel on side
    img[125:720,1216:1280] = sidebar
    #showing frame rate
    Ctime = time.time()
    fps = 1/(Ctime - Ptime)
    Ptime = Ctime
    cv2.putText(img,str(int(fps)),(100,700),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    #quit instructions
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("canvas",imgCanvas)
    cv2.imshow("detection App (press q to exit)", img)
    key = cv2.waitKey(1)
    
    #key q pressed quit
    if(key == 81 or key == 113):
        break
    
#release the webcam
cap.release()
cv2.destroyAllWindows()

print("completed in success")