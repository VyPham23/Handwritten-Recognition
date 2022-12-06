from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

#model = tf.keras.models.load_model('mnist.h5')
model = tf.keras.models.load_model('doodler.h5')
model.make_predict_function()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods = ['POST'])
def recognize():

    if request.method == 'POST':
        print('Receive image and predict what it is')
        data = request.get_json()
        imageBase64 = data['image']
        imgBytes = base64.b64decode(imageBase64)

        with open("temp.jpg", "wb") as temp:
            temp.write(imgBytes)

        with open('class_names.txt') as f:
            classes = f.readlines()
        classes = [c.replace('\n', '').replace(' ', '_') for c in classes]

        image = cv2.imread('temp.jpg')
        image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_prediction = np.reshape(image_gray, (28,28,1))
        image_prediction = (255 - image_prediction.astype('float')) / 255

        #prediction = np.argmax(model.predict(np.array([image_prediction])), axis = -1)

        prediction = model.predict(np.expand_dims(image_prediction, axis=0))[0]
        ind = (-prediction).argsort()[:1]
        latex = [classes[x] for x in ind]

        return jsonify({
            'prediction': str(latex),
            'status': True
        })


# app.run(host='0.0.0.0', port=81)

if __name__ == '__main__':
    app.run(debug=True)

# print(tf.__version__)