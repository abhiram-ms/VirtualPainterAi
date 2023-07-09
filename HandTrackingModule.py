import cv2
import time
import mediapipe as mp


class handDetector():
    def __init__(self, mode = False, maxHands =2,model_complexity = 1,detectionConfidence = 0.5, trackingConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence
        
        self.mpHands = mp.solutions.hands #MP MEANS MEDIA PIPE FROM IMPORTED
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.model_complexity, self.detectionConfidence,self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.tipIds = [4,8,12,16,20]
        
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #CONVERTING IMAGES INTO COLOR RGB 
        self.results = self.hands.process(imgRGB)  #CALLING MEDIAPIPE TO PROCESS

        if self.results.multi_hand_landmarks:  #CHECKING FOR MULTIPPLE HANDS
            for handLms in self.results.multi_hand_landmarks:    # FOR EACH HAND
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)           
        return img    
    
    
    def findPosition(self,img,handNo = 0,draw = True):
        self.lmList = []
        if self.results.multi_hand_landmarks: 
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):  #GETTING ID AND POSITION OF LANDMARKS
                h,w,c = img.shape                      #GETTING HEIGHT WIDTH CHANNEL OF OUR IMG TO CONVERT
                cx,cy = int(lm.x*w),int(lm.y*h)        #POSITIONS WE GOT FROM THE FOR LOOP INTO PIXELS
                # print(id,cx,cy)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)   #CIRCLE IN ID 0
        return self.lmList
    
    def fingersUp(self):
        fingers = []
        
        #thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        #4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
        

def main():
    detector = handDetector()
    cap = cv2.VideoCapture(0) #CAPTURING VIDEO
    
    Ptime = 0 #PREVIOUS TIME
    Ctime =0 #CURRENT TIME

    while True:
        success,im = cap.read() #VIDEO READING AS IMAGES
        im = cv2.resize(im,(1280,720))
        img = cv2.flip(im,1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[8])
        
        Ctime = time.time()
        fps = 1/(Ctime - Ptime)
        Ptime = Ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("detection App (press q to exit)", img)
        key = cv2.waitKey(1)
        
        #key q pressed quit
        if(key == 81 or key == 113):
            break


if __name__ == "__main__":
    main()





