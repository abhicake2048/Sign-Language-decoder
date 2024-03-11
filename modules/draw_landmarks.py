import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def get_results(model,frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Processing the image with holistic
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results,image

def draw_land(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                       mp_drawing.DrawingSpec(color=(0, 255, 123), thickness=1, circle_radius=1),
                                       mp_drawing.DrawingSpec(color=(13, 255, 222), thickness=1, circle_radius=1))

            # Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                               mp_drawing.DrawingSpec(color=(13, 255, 222), thickness=1, circle_radius=1))
    # Left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                               mp_drawing.DrawingSpec(color=(13, 255, 222), thickness=1, circle_radius=1))
    # Pose Detection
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(0, 0, 123), thickness=1, circle_radius=1),
                                       mp_drawing.DrawingSpec(color=(13, 255, 222), thickness=1, circle_radius=1))
    
def get_landmarks(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])

