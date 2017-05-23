# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from PIL import Image

from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data('F:\课程\机器学习\CIFAR\Convnet')
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X_predict = X_test[1:3]
y_predict = Y_test[1:3]

print(X_predict)


print(y_predict)
for j in range(0, len(y_predict[0])):
	if y_predict[0][j] == 1.0:
		print('Correct Class : ' + classes[j])
for j in range(0, len(y_predict[0])):
	if y_predict[1][j] == 1.0:
		print('Correct Class : ' + classes[j])



img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)


network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=1)
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='cifar10_cnn')

y_array = model.predict(X_predict)
for j in range(0, len(y_array[0])):
	if y_predict[0][j] == 1.0:
		print('Predict Class : ' + classes[j])
for j in range(0, len(y_predict[0])):
	if y_predict[1][j] == 1.0:
		print('Predict Class : ' + classes[j])