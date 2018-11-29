import numpy as np
import cv2
import imutils
from threading import Thread
from time import sleep

class FaceDetector(object):

    def __init__(self):
        self.fc = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        for a in range(10):
            ret, frame = self.cap.read()
        self.frame = self.modFrame(frame)
        gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        self.faces = self.fc.detectMultiScale(gray, 1.3, 5)
        self.running = True
        
        self.mid_x = np.size(self.frame,1) / 2
        self.mid_y = np.size(self.frame,0) / 3  
        self.sh = np.size(self.frame,0) / 8

        self.kp = 1 / self.mid_x
        self.kd = .1 / self.mid_x

        self.last_x = 0
        self.last_y = 0
        self.last_h = 0

        self.disp_thread = Thread(target=self.display)
        self.dtct_thread = Thread(target=self.detect)

        self.dtct_thread.start()
        self.display()

        cv2.destroyAllWindows()
        self.cap.release()

    def modFrame(self,frame):
        frame = cv2.blur(frame,(2,2))
        return frame

    def pd(self,x,y,h):
            ux = self.kp*(x - self.mid_x) + self.kd*(x - self.last_x)
            uy = self.kp*(y - self.mid_y) + self.kd*(y - self.last_y)
            uz = self.kp*(h - self.sh) + self.kd*(h - self.last_h)
            if ux > .4:
                ux = .4
            elif ux < -.4:
                ux = -.4
            if uy > .4:
                uy = .4
            elif uy < -.4:
                uy = -.4
            if uz > .3:
                uz = .3
            elif uz < -.3:
                uz = -.3
            self.last_x = x
            self.last_y = y
            self.last_h = h
            return(ux,uy,uz)

    def display(self):
        try:
            while self.running:
                ret, img = self.cap.read()
                self.frame = self.modFrame(img)
                for (x,y,w,h) in self.faces:
                    cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)
                    print(self.pd(x+w/2.0,y,h))
                cv2.imshow('frame',self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
        except Exception as e:
            print(e)

    def detect(self):
        while self.running:
            gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
            self.faces = self.fc.detectMultiScale(gray, 1.3, 5)


face = FaceDetector()