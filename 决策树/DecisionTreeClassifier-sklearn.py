import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
if __name__ == '__main__':
    print('Start read data...')
    time_1 = time.time()
    raw_data = pd.read_csv('E:\\lihang_algorithms-master\\data\\train.csv',header = 0)
    data = raw_data.values
    features = data[::,1:]
    labels = data[::,0]

    train_features , test_fearures ,train_labels , test_labels = train_test_split(features, labels,test_size=0.33,random_state=0)
    time_2 = time.time()
    print('read data cost %f second' % (time_2 -  time_1))
    print('Start training...')
    clf = DecisionTreeClassifier(criterion= 'entropy')# criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）
    clf.fit(train_features,train_labels)
    time_3 = time.time()
    print('Train data cost %f second' % (time_3 - time_2))
    test_predict = clf.predict(test_fearures)
    time_4 = time.time()
    print('predict cost %f second' % (time_4  -time_3))
    score =  accuracy_score(test_predict,test_labels)
print('The accuracy score is %f'% score)
