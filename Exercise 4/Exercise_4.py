import numpy as np
import pickle
from tensorflow import keras 
from tensorflow.keras.utils import to_categorical 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
PATH = '../input/cifar10/cifar-10-batches-py/'
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def onehot_encode(integer_encoded, n):
    onehot_encoded = list()
    integer_encoded=integer_encoded.astype(np.int64)
    for value in integer_encoded:
        listofzeros = [0] * n
        listofzeros[value] = 1
        onehot_encoded.append(listofzeros)
    onehot_encoded = np.array(onehot_encoded, dtype=float)
    return onehot_encoded

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


x_train,y_train,x_test,y_test = get_dataset(PATH)

#normalizing inputs from 0-255 to 0.0-1.0 
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 

x_test = x_test / 255.0
x_train = x_train / 255.0

# one hot encode outputs 
y_train =onehot_encode(y_train, 10) 
y_test = onehot_encode(y_test, 10)


x_train=x_train.reshape(-1, 3, 32, 32).transpose(0,2,3,1)
x_test= x_test.reshape(-1, 3, 32, 32).transpose(0,2,3,1)
num_classes = y_test.shape[1]


# Create the model 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Flatten()) 
model.add(Dropout(0.2)) 
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())


# Compile model 
lrate = 0.001 
epochs = 200
decay = lrate/epochs 
sgd = keras.optimizers.Adam(lr=lrate) 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#data augmentation
datagen = ImageDataGenerator(
                            rotation_range=15,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=False
                            )
datagen.fit(x_train)


rlr = ReduceLROnPlateau(monitor='val_accuracy', mode ='max', factor=0.5, min_lr=1e-7, verbose = 1, patience=3)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose = 1, patience=8, restore_best_weights=True)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose = 1, save_best_only=True)
callback_list = [rlr, es, mc]

history = model.fit(datagen.flow(x_train, y_train, batch_size = 64),
                                 validation_data = (x_test, y_test),
                                 epochs = 200, verbose = 1,
                                 callbacks = callback_list)
model.save('./best_model.h5')


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

 # Final evaluation of the model 
saved_model = load_model('./best_model.h5')
test_loss, test_acc  = model.evaluate(x_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (test_acc*100))


# compairing 1nn, bayes and cnn performancecr
data = {'1NN':42.26, 'Bayes':43.31, 'Neural Networks':85.08}
models = list(data.keys())
accuracies = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(models, accuracies,width = 0.4)
 
plt.xlabel("Classifiers")
plt.ylabel("Accuracy Obtained")
plt.title("Classifiers Comparison")
plt.ylim(0, 100)
plt.show()


