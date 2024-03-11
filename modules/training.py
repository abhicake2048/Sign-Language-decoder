from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.callbacks import TensorBoard


def get_x_y(actions,label_map,no_seq=30,seq_len=30,d_path="MP_DATA"):
    sequence,label = [],[]
    for action in actions:
        for seq in range(no_seq):
            window = []
            for frame_num in range(seq_len):
                res = np.load(os.path.join(d_path,action,str(seq),"{}.npy".format(frame_num)))
                window.append(res)
            sequence.append(window)
            label.append(label_map[action])
    return sequence,label

def train_model(x,y,actions):
    X = np.array(x)
    Y = to_categorical(y).astype(int)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.05)

    model = Sequential()
    model.add(LSTM(64,return_sequences=True,activation='relu',input_shape = (30,1662)))
    model.add(LSTM(128,return_sequences=True,activation='relu'))
    model.add(LSTM(64,return_sequences=False,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(actions.shape[0],activation='softmax'))

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    model.fit(X_train,y_train,epochs=300,callbacks=[tb_callback])

    return model




