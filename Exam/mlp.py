import numpy as np
from tensorflow import keras 
from tensorflow.keras.utils import to_categorical 

EPOCHS = 100
LR = 0.01

def load_dataset():
    X_test = np.loadtxt('X_test.txt').astype(np.float32)
    y_test = np.loadtxt('y_test.txt')
    X_train = np.loadtxt('X_train.txt').astype(np.float32)
    y_train = np.loadtxt('y_train.txt')
    return X_test, y_test, X_train, y_train

def train():
    # Load dataset
    X_test, y_test, X_train, y_train = load_dataset()
    print(X_train.shape)

    # One hot encode the labels
    y_train = to_categorical(y_train)
    print(y_train.shape)
    y_test = to_categorical(y_test)

    # Create the model
    model = mlp_model()

    # Train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print('Accuracy: %f' % (accuracy*100)) 


def mlp_model():
    model = keras.Sequential()
    # Input layer
    model.add(keras.layers.Dense(10, activation='relu', input_shape=(4,)))

    # Hidden layer
    model.add(keras.layers.Dense(10, activation='relu'))

    # Output layer
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    train()