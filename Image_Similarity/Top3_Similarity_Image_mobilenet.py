import os
from random import *
import numpy as np
import matplotlib.pyplot as plt

import PIL.Image as Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image

def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
# choose path
path = str(input('Enter The User Image Path: '))
file_names = os.listdir(path)
file_names = [x for x in file_names if 'jpg' in x]
file_names.sort()
# print('User_Num: {}'.format(path.split('/')[-2]))
# print('The Number Of Images: {}\n'.format(len(file_names)))
# pre processing
y_test = []
x_test = []

for file_name in file_names:
    abs_file_path = path + file_name
    img = image.load_img(abs_file_path, target_size=(224, 224))
    y_test.append(file_name.split('.')[0])
    # numpy로 바꾸면 빠름.
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if len(x_test) > 0:
        x_test = np.concatenate((x_test, x))
    else:
        x_test = x

x_test = tf.keras.applications.mobilenet.preprocess_input(x_test)
# modeling
model = tf.keras.applications.MobileNet(input_shape = None,
                                        include_top = False,
                                        weights = 'imagenet')

features = model.predict(x_test)

features_compress = features.reshape(len(y_test), 7 * 7 * 1024)
cos_sim = cosine_similarity(features_compress)
# print result
n = randint(0, len(y_test))
dct = {i:j for i, j in enumerate(y_test)}
_or = dct[n]
inputNos = np.array([n])

for inputNo in inputNos:
    top = np.argsort(-cos_sim[inputNo], axis=0)[1:4]
    sim = sorted(cos_sim[4], reverse = True)[1:4]
    recommend = [y_test[i] for i in top]
    _rec = []
    for r in recommend:
        _rec.append(r)

print ('=' * 150)

plt.figure(figsize = (15,5))

plt.subplot(141)
img = image.load_img(path + _or + '.jpg')
x = image.img_to_array(img)
x = x.astype('float32') / 255
plt.title('User_Food', fontsize = 15)
plt.imshow(x)

plt.subplot(142)
img = image.load_img(path + _rec[0] + '.jpg')
x = image.img_to_array(img)
x = x.astype('float32') / 255
plt.title('similarity_TOP_01\nScore: {0}'.format(str(sim[0])), fontsize = 15)
plt.imshow(x)

plt.subplot(143)
img = image.load_img(path + _rec[1] + '.jpg')
x = image.img_to_array(img)
x = x.astype('float32') / 255
plt.title('similarity_TOP_02\nScore: {0}'.format(str(sim[1])), fontsize = 15)
plt.imshow(x)

plt.subplot(144)
img = image.load_img(path + _rec[2] + '.jpg')
x = image.img_to_array(img)
x = x.astype('float32') / 255
plt.title('similarity_TOP_03\nScore: {0}'.format(str(sim[2])), fontsize = 15)
plt.imshow(x)


plt.show()
