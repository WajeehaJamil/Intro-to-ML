import numpy as np
from scipy.spatial.distance import cdist

def cifar_1nn():
    X_test, y_test, X_train, y_train = load_dataset()
    pred_1nn = cifar10_classifier_1nn(X_test, X_train, y_train)
    print("Accuracy with Cifar10 1NN Classifier: "+ str(class_acc(pred_1nn, y_test)))

def load_dataset():
    X_test = np.loadtxt('X_test.txt')
    y_test = np.loadtxt('y_test.txt')
    X_train = np.loadtxt('X_train.txt')
    y_train = np.loadtxt('y_train.txt')
    return X_test, y_test, X_train, y_train
    
def class_acc(pred,gt):
    accuracy = (sum(pred == gt)/len(pred))*100
    return accuracy

def cifar10_classifier_1nn(x_test,x_train,y_train):
    distance = cdist(x_test, x_train,'correlation')
    min_index = np.argmin(distance,axis=1)
    pred = y_train[min_index]
    return pred

if __name__ == '__main__':
    cifar_1nn()