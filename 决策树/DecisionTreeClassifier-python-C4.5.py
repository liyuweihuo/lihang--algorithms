import cv2
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class Tree():
    def __init__(self,note_type , Class = None , feature = None):
        self.note_tpe = note_type
        self.Class = Class
        self.feature = feature
        self.dict = {}
    def add_tree(self , key , tree):
        self.dict[key] = tree
    def predict(self , features):
        if self.note_tpe == 'leaf' or (features[self.feature] not in self.dict):
            return self.Class
        tree = self.dict.get(features[self.feature])
        return tree.predict(features)
def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0])/x.shape[0]
        logp = np.log2(p)
        ent-=p*logp
    return ent
def calc_condition_ent(x,y):
    x_value_list = set(x[i] for i in range(x.shape[0]))
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent+=(float(sub_y.shape[0])/y.shape[0])*temp_ent
    return ent
def calc_ent_grap(x,y):
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x,y)
    ent_grap = base_ent - condition_ent
    return ent_grap
def recures_train(train_set , train_label , features):
    LEAF = 'leaf'
    INTERNAL = 'internal'
    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(LEAF , Class = label_set.pop())
    class_len = [(i,len(list(filter(lambda x:x == i ,train_label)))) for i in range(class_num)]
    (max_class , max_len) = max(class_len , key=lambda  x:x[1])
    if len(features) == 0:
        return Tree(LEAF,Class = max_class)
    max_feature = 0
    max_gda = 0
    D = train_label
    for feature in features:
        A = np.array(train_set[:,feature].flat)
        gda = calc_ent_grap(A,D)
        if calc_ent(A)!=0:
            gda = gda/calc_ent(A)
        if gda>max_gda:
            max_gda = gda
            max_class = feature
    if max_gda < epsilon:
        return Tree(LEAF,Class = max_class)
    sub_features = list(filter(lambda x:x!=max_feature,features))
    tree = Tree(INTERNAL,feature = max_feature)
    max_feature_col = np.array(train_set[:,max_feature].flat)
    feature_value_list = set(max_feature_col[i] for i in range(max_feature_col.shape[0]))
    for feature_value in feature_value_list:
        index = []
        for i in range(len(train_label)):
            if train_set[i][max_feature] == feature_value:
                index.append(i)
        sub_train_set = train_set[index]
        sub_train_label = train_label[index]
        sub_tree = recures_train(sub_train_set,sub_train_label,sub_features)
        tree.add_tree(feature_value,sub_tree)
    return tree
def train(train_set , train_label , features):
    return recures_train(train_set,train_label,features)
def predict(test_set , tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)
class_num = 10
feature_len = 784
epsilon = 0.001
if __name__ == '__main__':
    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('E:\\lihang_algorithms-master\\data\\train.csv',header = 0)
    data = raw_data.values

    features = data[::,1::]
    #features = binaryzation_features(imgs)
    labels = data[::,0]

    train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size = 0.33,random_state = 0)
    time_2 = time.time()
    print('read data cost %f second' % (time_2 - time_1))

    print('Start training...')
    tree = train(train_features, train_labels, list(range(feature_len)))
    time_3 = time.time()
    print('training cost %f second ' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = predict(test_features,tree)
    time_4 = time.time()
    print('predicting cost %f second' % (time_4 - time_3))

    for i in range(len(test_predict)):
        if test_predict[i] ==None:
            test_predict[i] = epsilon
    score = accuracy_score(test_labels ,test_predict)
print('The accruacy score is %f' % score)