import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from model import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/')
def index():
    return render_template('index.html', label='')

# Upload endpoint (user presses "Capture" button)
@app.route('/upload', methods=['POST'])
def upload():
    # Gets the image
    if 'image' in request.files:
        image = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
        image.save(image_path)
        # Image is saved as "capture.png"
        # Load torch model
        prediction = predict(image_path)
        return render_template('index.html', label=prediction)
    else:
        return 'No image uploaded.'

if __name__ == "__main__":
    app.run(debug=True)
