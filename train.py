import sys
import os
import random
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

from keras.optimizers import SGD, adadelta, adagrad, adam, adamax, nadam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras import backend as K


from UNet import UNet_Model

# from data_generator import batch_generator, kitti_batch_generator


LEARNING_RATE = 0.0001
BATCH_SIZE = 16
NUM_EPOCHS = 1000
NUM_CHANNELS = 3

LOAD_PRE_TRAINED_MODEL = True
DO_TESTING = True


# need to use this when I require batches from an already memory-loaded X, y
def np_array_batch_generator(X, y, batch_size):
    batch_i = 0
    while 1:
        if (batch_i+1)*batch_size >= len(X):
            yield X[batch_i*batch_size:], y[batch_i*batch_size:]
            batch_i = 0
        else:
            yield X[batch_i*batch_size:(batch_i+1)*batch_size], y[batch_i*batch_size:(batch_i+1)*batch_size]

def charbonnier(y_true, y_pred):
    return K.sqrt(K.square(y_true - y_pred) + 0.01**2)

def main():

    X_val = np.load("X_val_KITTI.npy").astype("float32") / 255.
    y_val = np.load("y_val_KITTI.npy").astype("float32") / 255.

    ##### MODEL SETUP #####

    model = UNet_Model(input_shape=(6, 128, 384))

    optimizer = adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    loss = charbonnier

    model.compile(loss=loss, optimizer=optimizer)



    ##### TRAINING SETUP #####
    logging.warning("USING loss AS MODEL CHECKPOINT METRIC, CHANGE LATER!")
    callbacks = [
        # ModelCheckpoint(filepath="./../model_weights/weights.hdf5", monitor='val_loss', save_best_only=True, verbose=1),
        ModelCheckpoint(filepath="./../model_weights/weights.hdf5", monitor='loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=1)
    ]

    if LOAD_PRE_TRAINED_MODEL:
        print()
        logging.warning("LOADING PRE-TRAINED MODEL WEIGHTS!")
        print()
        model.load_weights("./../weights.hdf5")
        callbacks.append(CSVLogger("stats_per_epoch.csv", append=True))
    else:
        callbacks.append(CSVLogger("stats_per_epoch.csv", append=False))


    # OPTION 1: train on batches from youtube-8m
    # BATCH_IMAGE_SIZE = "random"
    # BATCH_IMAGE_SIZE = (320, 640)
    # BATCH_IMAGE_SIZE = (128, 384)
    # print "Begin training..."
    # hist = model.fit_generator(
    #     generator=batch_generator(BATCH_SIZE, NUM_CHANNELS, BATCH_IMAGE_SIZE),
    #     samples_per_epoch=800,
    #     nb_epoch=NUM_EPOCHS,
    #     callbacks=callbacks,
    #     validation_data=(X_val, y_val),
    #     max_q_size=10,
    #     nb_worker=1,
    #     # nb_worker=cpu_count(),
    # )

    # OPTION 2: train on batches from kitti
    hist = model.fit_generator(
        generator=kitti_batch_generator(BATCH_SIZE),
        samples_per_epoch=800,
        nb_epoch=NUM_EPOCHS,
        callbacks=callbacks,
        validation_data=np_array_batch_generator(X_val, y_val, BATCH_SIZE),
        nb_val_samples=len(X_val),
        max_q_size=10,
        nb_worker=1,
        # nb_worker=cpu_count(),    # deprecated
    )

    # OPTION 3: train on some memory-loaded data
    # hist = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, callbacks=callbacks, validation_data=(X_val, y_val), shuffle=True)

if __name__ == '__main__':
    main()
