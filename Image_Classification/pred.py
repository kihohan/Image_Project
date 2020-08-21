import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def pred(img_path):
    model = load_model('감자탕_김밥_김치_족발_죽.h5') 
    img = image.load_img(img_path, target_size = (100, 100))
    v = image.img_to_array(img)
    v = np.expand_dims(v, axis=0)
    X = tf.keras.applications.mobilenet.preprocess_input(v)
    pred = model.predict(X)
    prob = np.max(pred)
    r = np.argmax(pred)
    mylist = ['감자탕','김밥','김치','족발','죽']
    dct = {k:v for k, v in enumerate(mylist)}
    print ('pred:',dct[r])
    print ('prob: {}%'.format(prob * 100))

v = str(input('iamge_path: '))
pred(v)
