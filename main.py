import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import FaceDetection.FaceDetectorMediapipe as FaceDetectorMediapipe
import FaceDetection.FaceDetectorBlob as FaceDetectorBlob


FaceDetectorMediapipe.main()    #faster
#FaceDetectorBlob.main()         #more accurate