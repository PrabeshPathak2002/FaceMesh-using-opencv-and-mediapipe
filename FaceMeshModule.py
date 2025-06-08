import cv2
import mediapipe as mp
import time
import tkinter as tk

def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height

class FaceMesh:
    def __init__(self, camera_index=0, video_path=None, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.facemesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        self.mpDraw = mp.solutions.drawing_utils
        self.pTime = 0
        self.results = None

    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            return None  

        #Resize to fit screen
        screen_w, screen_h = get_screen_size()
        h, w = img.shape[:2]
        scale = min(screen_w / w, screen_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))      

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.results = self.facemesh.process(imgRGB)
        
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                    self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                    self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))

        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if (cTime - self.pTime) != 0 else 0
        self.pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        return img

    def findPosition(self, img, faceNo=0, draw=True, landmark_ids=None):
        """
        Returns a list of (id, x, y) for specified landmark_ids.
        If landmark_ids is None, returns all landmarks.
        """
        lmList = []
        if self.results and self.results.multi_face_landmarks:
            if faceNo < len(self.results.multi_face_landmarks):
                myFace = self.results.multi_face_landmarks[faceNo]
                h, w, c = img.shape
                for id, lm in enumerate(myFace.landmark):
                    if (landmark_ids is None) or (id in landmark_ids):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append((id, cx, cy))
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    face_mesh = FaceMesh(video_path="FaceVideos/5.mp4")
    try:
        while True:
            img = face_mesh.process_frame()
            if img is None:
                break
            
            lmList = face_mesh.findPosition(img, faceNo=0, draw=True, landmark_ids=[0]) 

            cv2.imshow("Face Mesh", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        face_mesh.release()

if __name__ == "__main__":
    main()
