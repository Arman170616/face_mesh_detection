"""
Created on Fri Jul  9 11:25:12 2021

@author: pyarena
"""

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) #using webcam, it depends on your system 0 or 1
presentTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)




while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                
    
     
    
    # fps control
    currentTime = time.time()
    fps_rate = 1 / (currentTime - presentTime)
    presentTime = currentTime
    
    cv2.putText(img, f'fps:{int(fps_rate)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    
    
    cv2.imshow('Face Mesh Detection', img)
    cv2.waitKey(10)
    
cap.release()
cv2.destroyAllWindows()
