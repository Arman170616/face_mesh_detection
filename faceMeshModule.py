import cv2
import mediapipe as mp
import time


class faceMeshDetector():
    def __init__(self, staticMode = False, maxFaces=2, minDetectionCon=0.5, minDetectionCon=0.5):
        
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minDetectionCon = minDetectionCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces = 2)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        
        

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)    

    



def main():
    
    cap = cv2.VideoCapture("/home/pyarena/python/OpenCV/faceMash/mesh.mp4")
    presentTime = 0
    
    while True:
        success, img = cap.read()
        # fps control
        currentTime = time.time()
        fps_rate = 1 / (currentTime - presentTime)
        presentTime = currentTime
        
        cv2.putText(img, f'fps:{int(fps_rate)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        
        
        cv2.imshow('Face Mesh Detection', img)
        cv2.waitKey(10)

    

if __name__ == "__main__":
    
    main()
