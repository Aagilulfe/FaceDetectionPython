import numpy as np
import cv2
from imutils.video import FPS

def FaceDetectorHaar():
    face_cascade = cv2.CascadeClassifier('FaceDetection/haarcascades/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot open camera'

    #img = cv2.imread('resources/repine.jpg')
    #img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    
    fps_stat = FPS().start()    # start the fps measurements

    while True:
        success, frame = cap.read()
        if not success: #
            continue
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('FaceDetectorHaar',frame)
        
        fps_stat.update()
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Prints the FPS stats
    fps_stat.stop()
    print("FaceDetectorHaar:")
    print("Elapsed time: {:.2f}".format(fps_stat.elapsed()))
    print("FPS: {:.2f}\n".format(fps_stat.fps()))


if __name__ == '__main__':
    FaceDetectorHaar()