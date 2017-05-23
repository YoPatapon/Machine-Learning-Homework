#!/usr/bin/env python
#encoding:utf-8

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import glob
import os
import time

if __name__ == "__main__":
    t0 = time.time()
    clf_type = 'SOFTMAX'
    fds = []
    labels = []
    num = 0
    total = 0
    for feat_path in glob.glob(os.path.join('./data/features/train', '*.feat')):
        data = joblib.load(feat_path)
        fds.append(data[:-1])
        labels.append(data[-1])
    if clf_type is 'LIN_SVM':
        clf = LinearSVC()
        print "Training a Linear SVM Classifier."
        clf.fit(fds, labels)

        for feat_path in glob.glob(os.path.join('./data/features/test', '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1, -1))
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                num += 1
        rate = float(num)/total
        t1 = time.time()
        print 'The classification accuracy is %f'%rate
        print 'The cast of time is :%f'%(t1 - t0)

    elif clf_type is 'SVM':
        clf = SVC(C = 0.5, decision_function_shape = 'ovr', degree = 4)
        print "Training a SVM Classifier."
        clf.fit(fds, labels)

        for feat_path in glob.glob(os.path.join('./data/features/test', '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1, -1))
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                num += 1
        rate = float(num)/total
        t1 = time.time()
        print 'The classification accuracy is %f'%rate
        print 'The cast of time is :%f'%(t1 - t0)

    elif clf_type is 'SOFTMAX':
        clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, solver='sag', multi_class='multinomial', verbose=0, warm_start=False, n_jobs=1)
        print "Training a softmax Classifier."
        clf.fit(fds, labels)

        for feat_path in glob.glob(os.path.join('./data/features/test', '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1, -1))
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                num += 1
        rate = float(num)/total
        t1 = time.time()
        print 'The classification accuracy is %f'%rate
        print 'The cast of time is :%f'%(t1 - t0)



