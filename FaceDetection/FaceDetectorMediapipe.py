import cv2
import mediapipe as mp
import time
from imutils.video import FPS


class FaceDetector():
    
    """
    Class for face detection using the openCV Blob method.
    Draws bbox around faces with percent score and FPS.
    
    Params:
    minDetectionCon: float, minimum confidence for detection (between 0 and 1)
    """
    
    def __init__(self, minDetectionCon=0.7):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

        self.bbox_color = (0, 69, 255)      # orange color for bbox
        
        
    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                if detection.score[0] > self.minDetectionCon:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)

                    bboxs.append([id, bbox, detection.score])
                    if draw:
                        img = self.fancyDraw(img,bbox)

                        cv2.putText(img, f'{int(detection.score[0] * 100)}%',
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


def main():
    
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    
    fps_stat = FPS().start()    # start the fps measurements

    detection_flag = True
    stats_flag = True

    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
        cv2.imshow("FaceDetectorMediapipe", img)

        fps_stat.update()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            detection_flag = not(detection_flag)
        if key == ord('s'):
            stats_flag = not(stats_flag)
    
    # Prints the FPS stats
    fps_stat.stop()
    print("\nFaceDetectorMediapipe:")
    print("Elapsed time: {:.2f}".format(fps_stat.elapsed()))
    print("FPS: {:.2f}".format(fps_stat.fps()))



if __name__ == '__main__':  # if script executed directly
    main()