#!/ user/bin/cnv python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import csv
path = r'G:\下载的程序\Human-Activity-Recognition-master\data\train1.csv'
Data = np.recfromcsv(path,delimiter=',',names=True, dtype=None)
#from sklearn import datasets
#data_iris = datasets.load_iris()
X = Data.data
y = Data.target
from sklearn import preprocessing
X1 = preprocessing.scale(X,axis=0)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=1)
from sklearn.svm import SVC
clf = SVC()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    return acc_rate

# 得到训练集的预测类别
y_hat1 = clf.predict(x_train)
train_rate = show_accuracy(y_hat1,y_train,'训练集')
# 打印训练集的预测类别
print(y_hat1)
# 打印训练集的原始类别
print(y_train)
# 打印训练集的准确率
print('train:%.3f%%', train_rate)
# 得到测试集的预测类别
y_hat = clf.predict(x_test)
test_rate = show_accuracy(y_hat,y_test,'测试集')
# 打印测试集的预测类别
print(y_hat)
# 打印测试集的原始类别
print(y_test)
# 打印测试集的准确率
print('test：%.3f%%',test_rate)

