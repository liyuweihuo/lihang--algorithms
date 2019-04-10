import pandas as pd
import numpy as np
import time
from sklearn.naive_bayes import MultinomialNB #朴素贝叶斯法
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    print("Start read data...")
    time_1 = time.time()

    raw_data = pd.read_csv("E:\\lihang_algorithms-master\\data\\train.csv",header = 0)
    data  = raw_data.values

    features = data[::,1::]
    labels = data[::,0]

    time_2 = time.time()
    print("read data cost %f seconds"%(time_2 - time_1))

    train_features , test_features , train_labels , test_labels = train_test_split(features ,labels , test_size= 0.33 , random_state= 0)

    print("Start training...")

    clf = MultinomialNB(alpha=1) #sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None),alpha:如果发现拟合的不好，需要调优时，可以选择稍大于1或者稍小于1的数。fit_prior:表示是否要考虑先验概率，如果是false,则所有的样本类别输出都有相同的类别先验概率，否则可以自己用第三个参数class_prior输入先验概率
    #sklearn.naive_bayes.GaussianNB  适用连续变量情况
    #sklearn.naive_bayes.BernoulliNB  要求特征是离散的，且为布尔类型，即true和false，或者1和0
    clf.fit(train_features , train_labels)
    time_3 = time.time()
    print("train data cost %f seconds" % (time_3 - time_2))

    print("Start predict...")
    test_predict = clf.predict(test_features)
    time_4 = time.time()
    print("predicting cost %f seconds" %(time_4 - time_3))
    score = accuracy_score(test_labels , test_predict)
print("The accruacy score is %f" % score)