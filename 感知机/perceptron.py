import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron(object):
    def __init__(self):
        self.learning_step = 0.001
        self.max_iteration = 5000

    def train(self,features,labels):
        self.w = [0.0]*(len(features[0])+1)

        correct_count = 0
        while correct_count < self.max_iteration:

            index = np.random.randint(0,len(labels)-1)
            x = list(features[index])
            x.append(1.0)
            y = 2*labels[index] - 1
            wx = sum([self.w[j]*x[j] for j in range(len(self.w))])

            if wx*y > 0:
                correct_count +=1
                continue
            for i in range(len(self.w)):
                self.w[i]+=self.learning_step*(y*x[i])
    def predict_(self , x):
        wx = sum([self.w[j]*x[j] for j in range(len(self.w))])
        return int(wx>0)
    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels
if __name__ == "__main__":

    time_1 = time.time()
    print("Starting read data...")
    raw_data = pd.read_csv("E:\\lihang_algorithms-master\\data\\train_binary.csv",header=0)
    data = raw_data.values

    features = data[::,1::]
    labels = data[::,0]

    time_2= time.time()
    print("read data cost %f seconds"%(time_2 - time_1))

    train_features , test_features , train_labels , test_labels = train_test_split(features , labels , test_size=0.33, random_state=0)
    print("Start train data...")
    p = Perceptron()
    p.train(train_features,train_labels)
    time_3 = time.time()
    print("train cost %f seconds"%(time_3 - time_2))
    print("Start predicting")
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print("predicting cost %f seconds"%(time_4 - time_3))
    score = accuracy_score(test_labels , test_predict)
print("The accruacy score is %f"%score)
