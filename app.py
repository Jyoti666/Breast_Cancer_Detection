import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import cv2

app = Flask(__name__)

#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
    cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

    while True:
    # Read the frame
        _, img = cap.read()

    # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display
        cv2.imshow('img', img)

    # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

# Release the VideoCapture object
    cap.release()
    """
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 0:
        res_val = "breast cancer"
    else:
        res_val = "no breast cancer"
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))"""

if __name__ == "__main__":
    app.run()
