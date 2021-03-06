import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Predict(testset , trainset , train_labels):
    predict = []
    count = 0
    for test_vec in testset:
        count +=1
        print("the number of %d is predicting...."%count)
        knn_list = []
        max_index = -1
        max_dist = 0
        for i in range(k):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec - test_vec)
            knn_list.append((dist,label))

        for i in range(k,len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec - test_vec)

            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]
            if dist < max_dist:
                knn_list[max_index] = (dist,label)
                max_index = -1
                max_dist = 0
        class_total = k
        class_count = [0 for i in range(class_total)]
        for dist , label in knn_list:
            class_count[label]+=1
        mmax = max(class_count)
        for i in range(class_total):
            if mmax == class_count[i]:
                predict.append(i)
                break
    return np.array(predict)

k = 10

if __name__ == '__main__':
    print("Start read data")

    time_1 = time.time()

    raw_data = pd.read_csv('E:\\lihang_algorithms-master\\data\\train.csv',header = 0)
    data = raw_data.values

    features = data[: , 1:]
    labels = data[:,0]
    train_features , test_features, train_labels, test_labels = train_test_split(features,labels,test_size = 0.33, random_state = 0)

    time_2 = time.time()
    print("read data cost %f second"%(time_2-time_1))

    print('Start training')
    print('knn need not train')

    time_3 = time.time()
    print("training cost %f second"%(time_3-time_2))

    print('Start predicting')
    test_predict = Predict(test_features,train_features,train_labels)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels,test_predict)
print('The accruacy score is %f '%score)
