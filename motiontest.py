import numpy as np
import cv2
import imutils

def modFrame(frame):
    frame = imutils.resize(frame,width=600)
    frame = cv2.blur(frame,(3,3))
    return frame

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
tracker = cv2.TrackerCSRT_create()
for a in range(10):
    ret, frame = cap.read()
ret, frame = cap.read()
frame = modFrame(frame)
boundbox = cv2.selectROI('selector',frame)
tracker.init(frame,boundbox)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = modFrame(frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        boundbox = cv2.selectROI('selector',frame)
        tracker.init(frame, boundbox)
    (success, bbox) = tracker.update(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if success:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    # Our operations on the frame come here

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
