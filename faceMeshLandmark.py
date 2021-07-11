
import cv2
import mediapipe as mp

#Face Mesh

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

#Image

image = cv2.imread("/home/python/OpenCV/faceMash/image.jpg")
height, width, _ = image.shape
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Facial landmarks
result = face_mesh.process(image)


for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, 468):
        
        pt1 = facial_landmarks.landmark[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)
        
        cv2.circle(image, (x,y), 2, (100,100,0), -1)
        #cv2.putText(image, str(i), (x,y), 0,1,(0,0,0))



cv2.imshow("Face Mesh detection", image)
cv2.waitKey(0)
