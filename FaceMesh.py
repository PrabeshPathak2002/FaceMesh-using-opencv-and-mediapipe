import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = facemesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id, lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, faceLms, mpfacemesh.FACEMESH_TESSELATION,
                                  mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                  mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()