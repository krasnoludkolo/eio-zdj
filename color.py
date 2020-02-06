import os

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.layers import InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from tensorflow import keras

model = keras.models.load_model('model.model')

color_me = []
for filename in os.listdir('Test/'):
    color_me.append(img_to_array(load_img('Test/' + filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
color_me = color_me.reshape(color_me.shape + (1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = color_me[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    imsave("result_color/img_" + str(i) + ".png", lab2rgb(cur))
