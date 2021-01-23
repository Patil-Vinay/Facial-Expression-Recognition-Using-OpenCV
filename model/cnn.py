# CNN for 7-class emotion classification on 48x48 grayscale faces (FER2013).
# Four conv blocks (32-64-128-256 filters), each with two 3x3 convs + batchnorm,
# followed by two dense layers.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                     MaxPooling2D, Dropout, Flatten, Dense)
from tensorflow.keras.regularizers import l2

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def conv_block(x, filters, reg):
    for _ in range(2):
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    return x


def build_cnn(input_shape=(48, 48, 1), num_classes=len(EMOTIONS)):
    reg = l2(1e-4)

    inputs = Input(shape=input_shape)
    x = inputs
    for f in [32, 64, 128, 256]:   # 48 -> 24 -> 12 -> 6 -> 3
        x = conv_block(x, f, reg)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='emotion_cnn')
