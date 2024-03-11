import mediapipe as mp
import cv2
from modules import draw_landmarks as dl
from modules import data_collection as dc
from modules import training as tr
import numpy as np
import keras


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# actions for which data is to be collected
actions = ["Hello","iloveyou","thanks"]

# Creating folders to collect data
dc.folder_creation(actions)

# Creating numpy arrays for frames for each of actions
dc.data_creation(actions)

# Getting X and Y to train our model
label_map = {label:num for num , label in enumerate(actions)}

xd,yd = tr.get_x_y(actions,label_map)

# To train the model
action = np.array(actions)
model = tr.train_model(xd,yd,action)

#saving the weights

model.save('action.hs')

# Rendering the video feed and predicting the sign language conversation
model = keras.models.load_model('action.hs')

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

sequence =[]
sentence = []
threshold = 0.7

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        results,image = dl.get_results(holistic,frame)

        dl.draw_land(image,results)

        keypoints = dl.get_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        sequence = np.array(sequence)
        sequence = np.repeat(sequence, 30, axis=0)
        sequence = sequence.tolist()
        if len(sequence)==30:
            res = model.predict(np.expand_dims(sequence,axis=0))[0]
        res = model.predict(np.expand_dims(sequence,axis=0))[0]
        if res[np.argmax(res)] > threshold :
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]
        image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image,(0,0),(640,40),(255,117,16),-1)
        cv2.putText(image,' '.join(sentence),(3,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

        cv2.imshow("OpenCV feed",image)

        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()