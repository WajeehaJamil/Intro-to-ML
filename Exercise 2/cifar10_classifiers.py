import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import time
from scipy.spatial.distance import cdist

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Part 2
def class_acc(pred,gt):
    accuracy = (sum(pred == gt)/len(pred))*100
    return accuracy

# Part 3
def cifar10_classifier_random(x):
    pred = []
    for i in range(len(x)):
        pred.append(randint(0,9))
    pred = np.array(pred)
    return pred

# Part 4
def cifar10_classifier_1nn(x_test,x_train,y_train):
    distance = cdist(x_test, x_train,'correlation')
    min_index = np.argmin(distance,axis=1)
    pred = y_train[min_index]
    return pred

if __name__ == '__main__':
    data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
    data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
    data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
    data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
    data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')

    test_batch = unpickle('cifar-10-batches-py/test_batch')

    x_train = np.concatenate((data_batch_1["data"], data_batch_2["data"],data_batch_3["data"], data_batch_4["data"],data_batch_5["data"]))
    y_train = np.concatenate((data_batch_1["labels"], data_batch_2["labels"], data_batch_3["labels"], data_batch_4["labels"], data_batch_5["labels"]))

    x_test = np.array(test_batch["data"].astype("float"))
    y_test = np.array(test_batch["labels"])

    #labeldict = unpickle('cifar-10-batches-py/batches.meta')
    #label_names = labeldict["label_names"]

    #x_train=x_train.reshape(10, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    #x_test= x_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

    pred_rand = cifar10_classifier_random(x_test)
    
    print("Accuracy with Cifar10 Random Classifier: "+ str(class_acc(pred_rand, y_test)))

    start_time = time.time()
    pred_1nn = cifar10_classifier_1nn(x_test, x_train, y_train)
    current_time = time. time()
    print("Finished iterating in for 1NN Classifier: " + str(int(current_time - start_time)) + " seconds")
    print("Accuracy with Cifar10 1NN Classifier: "+ str(class_acc(pred_1nn, y_test)))
