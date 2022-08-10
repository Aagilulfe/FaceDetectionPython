import cv2
import numpy as np
import mediapipe as mp

# Configuration of face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

img = cv2.imread('resources/lena.jpg', cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
# Performance improvement
image.flags.writeable = False
results = face_mesh.process(image)

