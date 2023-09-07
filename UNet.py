
from keras.layers import Layer, Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, BatchNormalization, Activation,concatenate
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model


def UNet_Model(input_shape):
    inputs = Input(input_shape)
    print("rdgxhjgsvxwhdck_---------------")
    print(inputs.shape)
    conv1 = Convolution2D(32, 3, 3, activation='relu', padding='same')(inputs)
    print(conv1.shape)
    bn1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', padding='same')(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))

    conv2 = Convolution2D(64, 3, 3, activation='relu', padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', padding='same')(bn2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))

    conv3 = Convolution2D(128, 3, 3, activation='relu', padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', padding='same')(bn3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))

    conv4 = Convolution2D(256, 3, 3, activation='relu', padding='same')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(256, 3, 3, activation='relu', padding='same')(bn4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))

    conv5 = Convolution2D(512, 3, 3, activation='relu', padding='same')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', padding='same')(bn5)
    bn5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))

    conv5_2 = Convolution2D(512, 3, 3, activation='relu', padding='same')(pool5)
    bn5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', padding='same')(bn5_2)
    bn5_2 = BatchNormalization()(conv5_2)
    
    # up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    
    up5_2 = concatenate([UpSampling2D(size=(2, 2))(bn5_2), bn5], axis=1)
    conv6_2 = Convolution2D(512, 3, 3, activation='relu', padding='same')(up5_2)
    bn6_2 = BatchNormalization()(conv6_2)
    conv6_2 = Convolution2D(512, 3, 3, activation='relu', padding='same')(bn6_2)
    bn6_2 = BatchNormalization()(conv6_2)

    up6 = concatenate([UpSampling2D(size=(2, 2))(bn6_2), bn4], axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', padding='same')(up6)
    bn6 = BatchNormalization()(conv6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', padding='same')(bn6)
    bn6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(bn6), bn3], axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', padding='same')(up7)
    bn7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', padding='same')(bn7)
    bn7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(bn7), bn2], axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', padding='same')(up8)
    bn8 = BatchNormalization()(conv8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', padding='same')(bn8)
    bn8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(bn8), bn1], axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', padding='same')(up9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', padding='same')(bn9)
    bn9 = BatchNormalization()(conv9)

    conv10 = Convolution2D(input_shape[0]/2, 1, 1, activation='sigmoid')(bn9)

    model = Model(input=inputs, output=conv10)

    return model

