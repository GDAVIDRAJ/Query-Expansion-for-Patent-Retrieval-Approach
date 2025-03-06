import tensorflow as tf
from keras import layers, models
import numpy as np
from Evaluation import evaluation

def Model(X, Y, BS):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.summary()
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(Y.shape[-1]))
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])  #  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.fit(X, Y, epochs=10, batch_size=BS)  # , validation_data=(X, Y))
    pred = model.predict(X)
    return pred

def Model_CNN(train_data, train_target, BS=None, sol=None):
    IMG_SIZE = 32

    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    pred = Model(Train_X, train_target, BS)
    # pred[pred >= 0.5] = 1
    # pred[pred < 0.5] = 0
    Eval = evaluation(pred, train_target)
    return Eval[7], Eval[5], Eval[12]

