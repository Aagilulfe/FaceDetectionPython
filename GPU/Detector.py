import numpy as np
import cv2
# from imutils.video import FPS
import time

class Detector:
    def __init__(self, CLIENT_IP, CLIENT_PORT, local_display, flip, batch, use_cuda=False):

        #read model
        self.faceModel = cv2.dnn.readNetFromCaffe("models/res10_300x300_ssd_iter_140000.prototxt", 
        caffeModel="models/res10_300x300_ssd_iter_140000.caffemodel")

        #connection to the client
        self.CLIENT_IP = CLIENT_IP
        self.CLIENT_PORT = CLIENT_PORT

        #video stream process
        dispW= 640
        dispH= 480
        self.flip = flip
        self.batch = batch
        self.local_display = local_display

        #I/O pipelines
        #self.camSet = 0
        self.camSet='nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
        self.directToClient = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host='+self.CLIENT_IP+' port='+str(self.CLIENT_PORT+1),cv2.CAP_GSTREAMER,0, 20, (dispW, dispH), True)
        self.out = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host='+self.CLIENT_IP+' port='+str(self.CLIENT_PORT),cv2.CAP_GSTREAMER,0, 20, (dispW, dispH), True)

        #enable cuda if required
        if use_cuda:
            print("Using CUDA")
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]

        self.processFrame()

        cv2.imshow("Output", self.img)
        # cv2.waitKey(0)

    def processVideo(self):
        cap = cv2.VideoCapture(self.camSet)
        # cap = cv2.VideoCapture(videoName)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return
        (success, self.img) = cap.read()
        (self.height, self.width) = self.img.shape[:2]

        self.directToClient.write(self.img)

        # fps = FPS().start()

        before = time.time()
        fps_manual = 20   # arbitrary fps to intialize the fps_manual
        flag = True
        count_batch = 0
        count_fps = 0

        while success:
            self.processFrame(count_batch)

            if count_fps == 40: # update fps_manual every 40 frames
                fps_manual = int(40/(time.time()-before))
                before = time.time() # reset timer
                count_fps = 0

            count_batch += 1
            count_fps += 1

            if count_batch == self.batch:
                count_batch = 0

            if flag:
                cv2.putText(self.img, "FPS: {}".format(fps_manual), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(self.img, "batch: {}".format(self.batch), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(self.img, "flip: {}".format(self.flip), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            if self.local_display:
                cv2.imshow("Output", self.img)
            self.out.write(self.img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                flag = not flag
            if self.batch != 1 and key == ord('a'):
                self.batch -= 1
                count_batch = 0
            elif self.batch != 20 and key == ord('z'):
                self.batch += 1
                count_batch = 0

            # fps.update()
            (success, self.img) = cap.read()
        
        # fps.stop()
        # print("Elapsed time: {:.2f}".format(fps.elapsed()))
        # print("FPS: {:.2f}".format(fps.fps()))

        cap.release()
        cv2.destroyAllWindows()

    def processFrame(self, count): # detect faces with blob and draw bounding box
        if count == 0:
            blob = cv2.dnn.blobFromImage(self.img, 1.0, (300, 300), (104.0, 177.0, 123.0), 
            swapRB=False, crop=False)

            self.faceModel.setInput(blob)
            self.predictions = self.faceModel.forward()

        for i in range(self.predictions.shape[2]):
            score = self.predictions[0, 0, i, 2]
            if score > 0.5:
                bbox = self.predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int")

                cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)