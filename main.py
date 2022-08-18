# FaceDetectionPython
#####################################################################################

import GPU

import FaceDetection.FaceDetectorHaar as FaceDetectorHaar
import FaceDetection.FaceDetectorBlob as FaceDetectorBlob
import FaceDetection.FaceDetectorMediapipe as FaceDetectorMediapipe

import FaceFilterMediapipe.apply_filter as apply_filter

#####################################################################################

FaceDetectorHaar.FaceDetectorHaar()         #basic method
FaceDetectorBlob.main()                     #more accurate
FaceDetectorMediapipe.main()                #faster

#apply_filter.apply_filter()