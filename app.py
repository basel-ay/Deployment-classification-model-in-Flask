from re import S
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

app = Flask(__name__, template_folder='templates')
model = tf.keras.models.load_model('model.h5')

dic = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Suprise'}

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(48,48))
	i = image.img_to_array(i)
	i = i.reshape(1, 48,48,3)
	predictions = model.predict(i)
    

	return predictions

    
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    img = request.files['img']
    img_path = img.filename	
    img.save(img_path)
    predictions = predict_label(img_path)
    y_pred = [np.argmax(probas) for probas in predictions]
    
    return render_template("index.html", prediction_text='Face should be : {}'.format(dic[y_pred[0]]), img_path = img_path)    
    
    
    
    
if __name__ == "__main__":
    app.run(debug=True)    