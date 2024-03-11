import numpy as np
import os
from modules import draw_landmarks as dl
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def folder_creation(actions):
    d_path = os.path.join("MP_DATA")

    no_seq = 30

    seq_len = 30

    for action in actions:
        for seq in range(no_seq):
            try:
                os.makedirs(os.path.join(d_path,action,str(seq)))
            except:
                pass
    
def data_creation(actions,no_seq=30,seq_len=30,d_path="MP_DATA"):
    
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for seq in range(no_seq):
                for frame_num in range(seq_len):
                    ret, frame = cap.read()

                    results,image = dl.get_results(holistic,frame)

                    dl.draw_land(image,results)

                    if frame_num==0:
                        cv2.putText(image,"STARTING COLLECTION",(120,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
                        cv2.putText(image,"Collecting frames for {} Video Number {}".format(action,seq),(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

                        cv2.imshow("OpenCV feed", image)
                        cv2.waitKey(900)
                    else:
                        cv2.putText(image,"Collecting frames for {} Video Number {}".format(action,seq),(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

                        cv2.imshow("OpenCV feed", image)
                    
                    keypoints = dl.get_landmarks(results)
                    npy_path = os.path.join(d_path,action,str(seq),str(frame_num))
                    np.save(npy_path,keypoints)


                    if cv2.waitKey(10) & 0xFF == ord('x'):
                        break

        cap.release()
        cv2.destroyAllWindows()
