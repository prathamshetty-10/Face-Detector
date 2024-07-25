import cv2
import cv2.data
import numpy 

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")

def detect_features(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,minNeighbors=5)

    for (x,y,w,h) in faces:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=5)
        face=frame[y:y+h,x:x+w]
        gray_face=gray[y:y+h,x:x+w]

        smiles=smile_cascade.detectMultiScale(gray_face,2.5,minNeighbors=9)
        for (xp,yp,wp,hp) in smiles:
            face=cv2.rectangle(face,(xp,yp),(xp+wp,yp+hp),color=(0,0,255),thickness=5)

        eyes=eye_cascade.detectMultiScale(gray_face,2.5,minNeighbors=5)
        for (xp,yp,wp,hp) in eyes:
            face=cv2.rectangle(face,(xp,yp),(xp+wp,yp+hp),color=(255,0,0),thickness=5)

    return frame

stream=cv2.VideoCapture(0)
if not stream.isOpened():
    print("No Cam")
    exit()

while(True):
    ret,frame=stream.read()
    if not ret:
        print("No more cam")
        break
    frame=detect_features(frame)
    cv2.imshow("Webcam",frame)
    if cv2.waitKey(1)==ord('q'):
        break

stream.release()
cv2.destroyAllWindows()