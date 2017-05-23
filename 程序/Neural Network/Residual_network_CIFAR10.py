# -*- coding: utf-8 -*-WW

from __future__ import division, print_function, absolute_import

import tflearn

n = 5

from tflearn.datasets import cifar10

(X, Y), (X_test, Y_test) = cifar10.load_data('F:\课程\机器学习\CIFAR\Convnet')

Y = tflearn.data_utils.to_categorical(Y, 10)
Y_test = tflearn.data_utils.to_categorical(Y_test, 10)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X_predict = X_test[9997:10000]
y_predict = Y_test[9997:10000]

print(X_predict)


print(y_predict)
for j in range(0, len(y_predict[0])):
  if y_predict[0][j] == 1.0:
    print('Correct Class : ' + classes[j])
for j in range(0, len(y_predict[0])):
  if y_predict[1][j] == 1.0:
    print('Correct Class : ' + classes[j])
for j in range(0, len(y_predict[0])):
  if y_predict[2][j] == 1.0:
    print('Correct Class : ' + classes[j])

img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16) 
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

net = tflearn.fully_connected(net, 10, activation = 'softmax')
mom = tflearn.Momentum(0.1, lr_decay = 0.1, decay_step = 32000, staircase = True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose = 1, clip_gradients = 0.)

model.fit(X, Y, n_epoch = 120, validation_set = (X_test, Y_test), snapshot_epoch = False, snapshot_step = 500,show_metric = True, batch_size = 128, shuffle = True, run_id = 'resnet_cifar10')

y_array = model.predict(X_predict)
for j in range(0, len(y_array[0])):
  if y_predict[0][j] == 1.0:
    print('Predict Class : ' + classes[j])
for j in range(0, len(y_predict[0])):
  if y_predict[1][j] == 1.0:
    print('Predict Class : ' + classes[j])
for j in range(0, len(y_predict[0])):
  if y_predict[2][j] == 1.0:
    print('Predict Class : ' + classes[j])