import cv2
import numpy as np
import time
from imutils.video import FPS


class FaceDetector():
    
    """
    Class for face detection using the openCV Blob method.
    Draws bbox around faces with percent score and FPS.
    
    Params:
    CLIENT_IP: ipv4 address
    CLIENT_PORT: port number
    use_cuda: boolean, if true, use cuda
    minDetectionCon: float, minimum confidence for detection (between 0 and 1)
    """
    
    def __init__(self, CLIENT_IP, CLIENT_PORT, use_cuda, minDetectionCon=0.7):

        #minimum of detection
        self.minDetectionCon = minDetectionCon

        #read model
        self.faceModel = cv2.dnn.readNetFromCaffe("FaceDetection/blob_models/res10_300x300_ssd_iter_140000.prototxt", 
        caffeModel="FaceDetection/blob_models/res10_300x300_ssd_iter_140000.caffemodel")

        #connection to the client
        self.CLIENT_IP = CLIENT_IP
        self.CLIENT_PORT = CLIENT_PORT

        #video stream process
        dispW= 640
        dispH= 480

        #I/O pipelines
        self.camSet='nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
        self.out = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host='+self.CLIENT_IP+' port='+str(self.CLIENT_PORT),cv2.CAP_GSTREAMER,0, 20, (dispW, dispH), True)

        #enable cuda if required
        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.bbox_color = (0, 69, 255)      # orange color for bbox
        
        
    def findFaces(self, img, draw=True):
       
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), 
        swapRB=False, crop=False)

        self.faceModel.setInput(blob)
        self.predictions = self.faceModel.forward()

        bboxs = []

        for i in range(self.predictions.shape[2]):
            score = self.predictions[0, 0, i, 2]

            if score > self.minDetectionCon:
                ih, iw, ic = img.shape
                bbox = int(self.predictions[0, 0, i, 3] * iw), int(self.predictions[0, 0, i, 4] * ih), \
                    int((self.predictions[0, 0, i, 5]-self.predictions[0, 0, i, 3]) * iw), int((self.predictions[0, 0, i, 6]-self.predictions[0, 0, i, 4]) * ih)
                
                bboxs.append([id, bbox, score])

                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.putText(img, f'{int(score * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            1.2, self.bbox_color, 2)
        
        return img, bboxs

    
    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, self.bbox_color, rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), self.bbox_color, t)
        cv2.line(img, (x, y), (x, y+l), self.bbox_color, t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), self.bbox_color, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), self.bbox_color, t)
        
        return img


def main(use_cuda=False):

    dispW= 640
    dispH= 480
    
    #camSet = 0
    camSet = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)15/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    
    cap = cv2.VideoCapture(camSet)
    assert cap.isOpened(), 'Cannot open camera'
    
    pTime = 0
    detector = FaceDetector("0.0.0.0", 5000, use_cuda)
    
    fps_stat = FPS().start()    # start the fps measurements

    detection_flag = True
    stats_flag = True
    
    while True:
        success, img = cap.read()
        if not success:
            continue
        
        if detection_flag:
            img, bboxs = detector.findFaces(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        if stats_flag:
            cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
        
        cv2.imshow("FaceDetectorBlob", img)

        fps_stat.update()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            detection_flag = not(detection_flag)
        if key == ord('s'):
            stats_flag = not(stats_flag)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Prints the FPS stats
    fps_stat.stop()
    print("FaceDetectorBlob:")
    print("Elapsed time: {:.2f}".format(fps_stat.elapsed()))
    print("FPS: {:.2f}\n".format(fps_stat.fps()))



if __name__ == '__main__':  # if script executed directly
    main()