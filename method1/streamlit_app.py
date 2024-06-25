import os
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd
import streamlit as st
model = load_model('Model/smnistkaggle.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils


# letter j  and z is not included due to gesture motion 
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

st.title("American Sign Language Detection")
frame_placeholder = st.empty()

user_input = st.text_input("Enter a video source url:")

url = 0
if user_input:
    st.write("You entered:", user_input)
    url = user_input
    #"http://100.115.192.101:8080/video"

cap = cv2.VideoCapture(url)
_, frame = cap.read()
h, w, c = frame.shape
print(h, w)
start_btn_pressed = st.button("start")
stop_btn_pressed = st.button("stop")

if start_btn_pressed:
    while True and not stop_btn_pressed:
        cap = cv2.VideoCapture(url)
        ret,frame = cap.read()
        analysisframe = frame
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
        # analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if hand_landmarksanalysis:
            print('Hand Detected')
            #Draw the landmarks in the frame
            # for landmarks in hand_landmarksanalysis:
            #     mp_drawing.draw_landmarks(analysisframe, landmarks, mphands.HAND_CONNECTIONS)

            # create a outer box around the hand
            for handLMsanalysis in hand_landmarksanalysis:
                x_min = w
                x_max = 0
                y_min = h
                y_max = 0

                for landmarks in handLMsanalysis.landmark:
                    x, y  = int(landmarks.x * w), int(landmarks.y * h)
                    # print(x,y)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 3)
                mp_drawing.draw_landmarks(frame, handLMsanalysis, mphands.HAND_CONNECTIONS)

            try:
                analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv2.resize(analysisframe,(28,28))
                flat_image = analysisframe.flatten()
                datan = pd.DataFrame(flat_image).T
                # datan = analysisframe.flatten()
                # datan = datan/255
                pixeldata = datan.values
                pixeldata = pixeldata / 255
                pixeldata = pixeldata.reshape(-1,28,28,1)

                # print(pixeldata.shape)

                #prediction
                prediction = model.predict(pixeldata)
                predarray = np.array(prediction[0])

                # print(predarray)

                letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
                letter,probabality = "",0

                for key,value in letter_prediction_dict.items():
                    if value > probabality:
                        probabality = value
                        letter = key

                letter = "{} prob:{}".format(letter, probabality)
                font = cv2.FONT_HERSHEY_SIMPLEX
                position = (x_max, y_min)  # Specify the (x, y) coordinates where you want to place the text
                font_scale = round(h/400)  # Font scale

                font_color = (255, 255, 255)  # Font color in BGR format (white in this example)
                font_thickness = round(h/200)  # Font thickness

                # Draw the text on the frame
                cv2.putText(frame,letter, position, font, font_scale, font_color, font_thickness)     
                print(font_scale,h)        
                print(letter, probabality)


            except cv2.error as e:
                # print("analysisframe is empty")
                pass

        # cv2.imshow("Frame", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()