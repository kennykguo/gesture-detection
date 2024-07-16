import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from keras.models import load_model
from spellchecker import SpellChecker
import threading

app = Flask(__name__, template_folder='templates')

# Load the pre-trained model and spell checker
model = load_model('model.h5')
spell = SpellChecker()

alphabet = {chr(i + 96).upper(): i for i in range(1, 27)}
alphabet['del'] = 27
alphabet['nothing'] = 28
alphabet['space'] = 29

# Thresholds for edge detection
lower_threshold = 100
upper_threshold = 0

IMG_SIZE = 100
THRESHOLD = 0.85
N_FRAMES = 70
SENTENCE = ''
LETTERS = np.array([], dtype='object')
START = False

camera_active = False
video_capture = None

def get_class_label(val, dictionary):
    for key, value in dictionary.items():
        if value == val:
            return key

def generate_frames():
    global START, SENTENCE, LETTERS, lower_threshold, upper_threshold, camera_active, video_capture
    while camera_active:
        if video_capture is None:
            continue
        
        ret, frame = video_capture.read()
        if not ret:
            break

        x_0 = int(frame.shape[1] * 0.1)
        y_0 = int(frame.shape[0] * 0.1)
        x_1 = int(x_0 + 200)
        y_1 = int(y_0 + 200)

        hand = frame[y_0:y_1, x_0:x_1]
        gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        blured = cv2.GaussianBlur(gray, (5, 5), 0)
        blured = cv2.erode(blured, None, iterations=2)
        blured = cv2.dilate(blured, None, iterations=2)

        edged = cv2.Canny(blured, lower_threshold, upper_threshold)
        model_image = ~edged
        model_image = cv2.resize(model_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        model_image = model_image.astype('float32') / 255.0
        model_image = model_image.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        predict = model.predict(model_image)
        for values in predict:
            if np.all(values < 0.5):
                letter = 'Cannot classify :('
            else:
                predict = np.argmax(predict, axis=1) + 1
                letter = get_class_label(predict, alphabet)
                LETTERS = np.append(LETTERS, letter)

        if START:
            if (np.mean(LETTERS[-N_FRAMES:] == letter) >= THRESHOLD) & (len(LETTERS) >= N_FRAMES):
                if letter == 'space':
                    SENTENCE = SENTENCE[:-1] + ' ' + '_'
                    LETTERS = np.array([], dtype='object')
                elif letter == 'del':
                    SENTENCE = SENTENCE[:-2] + '_'
                    LETTERS = np.array([], dtype='object')
                elif letter == 'nothing':
                    pass
                else:
                    SENTENCE = SENTENCE[:-1] + letter + '_'
                    LETTERS = np.array([], dtype='object')

            if len(SENTENCE) > 2 and SENTENCE[-3:] == '  _':
                SENTENCE = SENTENCE.split(' ')
                word = SENTENCE[-3]
                corrected_word = spell.correction(word)
                SENTENCE[-3] = corrected_word.upper()
                SENTENCE = ' '.join(SENTENCE[:-2]) + ' _'

        cv2.rectangle(frame, (x_0, y_0), (x_1, y_1), (0, 255, 0), 2)
        cv2.putText(frame, 'Place your hand here:', (x_0 - 30, y_0 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        cv2.putText(frame, letter, (x_0 + 10, y_0 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.putText(frame, 'Result: ' + SENTENCE, (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if video_capture is not None:
        video_capture.release()
        video_capture = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global lower_threshold, upper_threshold
    data = request.json
    lower_threshold = int(data.get('lower', lower_threshold))
    upper_threshold = int(data.get('upper', upper_threshold))
    return jsonify(status="success", lower_threshold=lower_threshold, upper_threshold=upper_threshold)

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, video_capture
    if not camera_active:
        camera_active = True
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return jsonify(status="success")

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify(status="success")

@app.route('/stop', methods=['POST'])
def stop():
    global camera_active
    camera_active = False
    shutdown_server()
    return jsonify(status="success")

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)