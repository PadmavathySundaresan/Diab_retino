import keras
import tensorflow as tf
import base64
import io
import numpy as np
import os
import pickle
from PIL import Image
from keras import backend as K
from flask import request
from flask import jsonify
from flask import Flask, render_template
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.image import img_to_array
import scipy

app = Flask(__name__)

def prediction_from_Inception(image):
    image_size1 = (299, 299)    
    img = image.resize(image_size1)
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)    
    x = preprocess_input(x)
    feature = model_test1.predict(x)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    preds = classifier1.predict(flat)
    print(preds)
    return preds

   

def prediction_from_MobileNet(image):
    image_size2 = (224, 224)      
    img = image.resize(image_size2)
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)    
    x = preprocess_input(x)
    feature = model_test2.predict(x)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    preds = classifier2.predict(flat)
    print(preds)
    return preds
   
   
   
def prediction_from_Xception(image):
    image_size3 = (299, 299)        
    img = image.resize(image_size3)
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)    
    x = preprocess_input(x)
    feature = model_test3.predict(x)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    preds = classifier3.predict(flat)
    print(preds)
    return preds
   

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

print("HI. WELCOME TO THE SERVER")
# global graph
# graph = tf.get_default_graph()
# global model
# print("MODEL LOADED!!!")
print('LOADING INCEPTION V3...')
classifier1 = pickle.load(open(r'classifier_lr_idird_final.cpickle', 'rb')) #inception V3
base_model1 = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(299,299,3)))
model_test1 = Model(input=base_model1.input, output=base_model1.layers[-1].output)
print('INCEPTION V3 MODEL LOADED!!')

print('LOADING MOBILENET...')
classifier2 = pickle.load(open(r'classifier_mobilenet.cpickle', 'rb'))
base_model2 = MobileNet(include_top=False, weights='imagenet', input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
model_test2 = Model(input=base_model2.input, output=base_model2.layers[-1].output)
print('MOBILENET MODEL LOADED!!')

print('LOADING XCEPTION...')
classifier3 = pickle.load(open(r'classifier1 (1).cpickle', 'rb'))
base_model3 = Xception(weights='imagenet')
model_test3 = Model(input=base_model3.input, output=base_model3.get_layer('avg_pool').output)    
print('XCEPTION MODEL LOADED!!')

@app.route("/predict", methods=["POST"])
def doPrediction():
    print("GOT REQUEST")
    message = request.get_json(force=True)
    #print(message)
    response = message['image']
    encoded = response[23:]
    decoded = base64.b64decode(encoded)
    dataBytesIO = io.BytesIO(decoded)
    image = Image.open(dataBytesIO)
   
    processed_image = preprocess_image(image)  
    pred1 = prediction_from_Inception(processed_image)
    pred2 = prediction_from_MobileNet(processed_image)
    pred3 = prediction_from_Xception(processed_image)
   
    labels=[pred1, pred2, pred3]
    print(labels)    
   
    labels = np.array(labels)
    labels = np.transpose(labels)
    labels = scipy.stats.mode(labels,axis=None)[0]
    labels = np.squeeze(labels)
    print(labels.tolist())
   
   
    response = {
        'predictions': {
            'grade' : labels.tolist()    
        }  
    }
    return jsonify(response)
