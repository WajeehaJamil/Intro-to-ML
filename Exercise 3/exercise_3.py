import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from skimage.transform import rescale, resize, downscale_local_mean
from math import exp
from scipy.stats import multivariate_normal

PATH = 'cifar-10-batches-py/'

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def class_acc(pred,gt):
    accuracy = (sum(pred == gt)/len(pred))*100
    return accuracy


def get_dataset(path):
    x_train = np.empty([50000, 3072]).astype(np.float64)
    y_train = np.empty([50000]).astype(np.float64)
    for i in range(5):   
        x_train[i*10000:((i+1)*10000)] = (list(unpickle(path+ 'data_batch_' +str(i+1))["data"]))
        y_train[i*10000:((i+1)*10000)] = (list(unpickle(path+ 'data_batch_' +str(i+1))["labels"]))
    
    x_test = np.empty([10000, 3072]).astype(np.float64)
    y_test = np.empty([10000]).astype(np.float64)
    x_test[:] = (list(unpickle(path+ 'test_batch')["data"]))
    y_test[:] = (list(unpickle(path+ 'test_batch')["labels"]))

    return x_train,y_train,x_test,y_test

def cifar10_n_color(X, dim):
    len_x = X.shape[0]
    
    # to convert the 3072 length vectors to images
    X=X.reshape(len_x, 3, 32, 32).transpose(0,2,3,1).astype("float32")
    X_mean = np.zeros((len_x,dim,dim,3)) 
    for i in range(X.shape[0]):
    # Convert images to mean values of each color channel
        img = X[i]      
        img_n = resize(img, (dim, dim))        
        r_vals = img_n[:,:,0]
        g_vals = img_n[:,:,1]
        b_vals = img_n[:,:,2]
        X_mean[i] = np.array((r_vals, g_vals, b_vals)).transpose(1,2,0)
  
    return X_mean


def cifar_10_n_bayes_learn(Xf,Y,dim):
    #Xf = Xf.transpose(0,3,1,2)
    data_r = dict()
    data_g = dict()
    data_b = dict()
    data = dict()
    Xf_sorted = np.arange(150000*dim*dim).reshape(10,5000,dim*dim*3).astype("float32")
    for i in range(0,len(Y)): 
        
        if Y[i] not in data_r:
            data_r[Y[i]] = np.array([])        
        if Y[i] not in data_g:
            data_g[Y[i]] = np.array([])       
        if Y[i] not in data_b:
            data_b[Y[i]] = np.array([])       
        if (Y[i] not in data):
            data[Y[i]] = np.array([])         
        data_r[Y[i]] = np.append(data_r[Y[i]],Xf[i][0])
        data_g[Y[i]] = np.append(data_g[Y[i]],Xf[i][1])
        data_b[Y[i]] = np.append(data_b[Y[i]],Xf[i][2])            
        data[Y[i]] = np.append(data[Y[i]],Xf[i])
        
    mean = np.arange(30*dim*dim).reshape(10,3*dim*dim).astype("float32")
    std = np.arange(10*(dim*dim*3)*(dim*dim*3)).reshape(10,dim*dim*3,dim*dim*3).astype("float32")
    p = np.full((10,1),0.1)
    
    for i in range(0,10):
        Xf_sorted[i] = data[i].reshape(5000,3*dim*dim)
        data_r[i] = data_r[i].reshape(5000,dim,dim)
        data_g[i] = data_g[i].reshape(5000,dim,dim)
        data_b[i] = data_b[i].reshape(5000,dim,dim)
        
        mean[i] = np.concatenate((np.mean(data_r[i],axis=0),np.mean(data_g[i],axis=0),np.mean(data_b[i],axis=0))).reshape(dim*dim*3)
        std[i] = np.cov(Xf_sorted[i],rowvar=False)

    return mean,std,p


def cifar_10_naivebayes_n_learn(Xf,Y,dim):
    
    data_r = dict()
    data_g = dict()
    data_b = dict()
    data = dict()
    Xf_sorted = np.arange(150000*dim*dim).reshape(10,5000,dim*dim*3).astype("float32")
    
    for i in range(0,len(Y)): 
        
        if Y[i] not in data_r:
            data_r[Y[i]] = np.array([])        
        if Y[i] not in data_g:
            data_g[Y[i]] = np.array([])       
        if Y[i] not in data_b:
            data_b[Y[i]] = np.array([])       
        if (Y[i] not in data):
            data[Y[i]] = np.array([])            
        data_r[Y[i]] = np.append(data_r[Y[i]],Xf[i][0])
        data_g[Y[i]] = np.append(data_g[Y[i]],Xf[i][1])
        data_b[Y[i]] = np.append(data_b[Y[i]],Xf[i][2])            
        data[Y[i]] = np.append(data[Y[i]],Xf[i])
        
    mean = np.arange(30*dim*dim).reshape(10,3*dim*dim).astype("float32")
    std = np.arange(30*dim*dim).reshape(10,3*dim*dim).astype("float32")
    
    p = np.full((10,1),0.1)    
    for i in range(0,10):
        data_r[i] = data_r[i].reshape(5000,dim,dim)
        data_g[i] = data_g[i].reshape(5000,dim,dim)
        data_b[i] = data_b[i].reshape(5000,dim,dim)
        
        mean[i] = np.concatenate((np.mean(data_r[i],axis=0),np.mean(data_g[i],axis=0),np.mean(data_b[i],axis=0))).reshape(dim*dim*3)  
      
        std[i] =  np.concatenate((np.std(data_r[i],axis=0),np.std(data_g[i],axis=0),np.std(data_b[i],axis=0))).reshape(dim*dim*3)  
      
    return mean,std,p


def cifar10_classifier_n_naivebayes(x, mu, sigma, p, dim):
    probabilities = np.ones(10)
    for i in range(10):
        for j in range(3*dim*dim):
            probabilities[i] *= norm.pdf(x[j], mu[i][j], sigma[i][j])
    return np.argmax(probabilities)

def cifar10_classifier_bayes(x,mu,sigma,p):
    probabilities = np.ones(10)
    for i in range(10):
        probabilities[i] *= multivariate_normal.logpdf(x, mu[i], sigma[i],allow_singular=True)
    return np.argmax(probabilities)

def plot(acc_b, acc_nb):
    x = np.array([1,2,4,8,16])
    my_xticks = ['1x1','2x2','4x4','8x8','16x16']
    plt.xticks(x, my_xticks)
    plt.plot(x ,acc_b, marker='o',markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label = "Bayes")
    plt.legend()
    plt.grid(True)
    plt.axis([0,20,0,60])

    for i,j in zip(x,acc_b):
        plt.annotate(str(j),xy=(i-0.25,j+4))

    plt.plot(x, acc_nb, marker='o',markerfacecolor='green', markersize=12, color='lightgreen', linewidth=4, label = 'Naive Bayes')
    plt.legend()
    for i,j in zip(x,acc_nb):
        plt.annotate(str(j),xy=(i-0.25,j-5))
    plt.xlabel('Image size')
    plt.ylabel('Accuracy (%)')
    plt.show()

if __name__ == '__main__':
    x_train,y_train,x_test,y_test = get_dataset(PATH)
    dimen = [1, 2, 4, 8, 16]
    accuracy_nb = []
    accuracy_b = []

    for j,dim in enumerate(dimen):
        print("="*30)
        x_train_mean = cifar10_n_color(x_train,dim)
        x_test_mean = cifar10_n_color(x_test,dim)
        print("Shape of x train mean for " + str(dim) + "x" + str(dim) + ":", np.shape(x_train_mean))
        print("Shape of x test mean for " + str(dim) + "x" + str(dim) +":", np.shape(x_test_mean))
        mu_bayes,sigma_bayes,p = cifar_10_n_bayes_learn(x_train_mean.transpose(0,3,1,2),y_train,dim)
        mu_naive,sigma_naive,p = cifar_10_naivebayes_n_learn(x_train_mean.transpose(0,3,1,2),y_train,dim)

        print('Shape of mu --- Naive Bayes:',np.shape(mu_naive),'Bayes:',np.shape(mu_bayes))
        print('Shape of sigma --- Naive Bayes:',np.shape(sigma_naive),'Bayes:',np.shape(sigma_bayes))
        print('Shape of p',np.shape(p))
        predicted_value_b =np.array([])
        predicted_value_nb =np.array([])

        for i in range(0,10000):
            predicted_value_b = np.append(predicted_value_b,  cifar10_classifier_bayes(x_test_mean[i].transpose(2,0,1).reshape(dim*dim*3),mu_bayes,sigma_bayes,p))
            predicted_value_nb = np.append(predicted_value_nb,  cifar10_classifier_n_naivebayes(x_test_mean[i].transpose(2,0,1).reshape(dim*dim*3),mu_naive,sigma_naive,p,dim))

        accuracy_nb.append(class_acc(predicted_value_nb,y_test))
        accuracy_b.append(class_acc(predicted_value_b,y_test))
        print("Accuracy for" + str(dim) + "x" + str(dim) + " Naive Bayes:" + str(accuracy_nb[j]) + "%")
        print("Accuracy for " + str(dim) + "x" + str(dim) +" Bayes:" + str(accuracy_b[j]) + "%")

    plot(accuracy_b,accuracy_nb)