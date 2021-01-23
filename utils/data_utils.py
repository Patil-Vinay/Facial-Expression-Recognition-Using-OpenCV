# Loading and preprocessing for the FER2013 dataset.
# Dataset: https://www.kaggle.com/datasets/msambare/fer2013 (fer2013.csv)

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.cnn import EMOTIONS

IMG_SIZE = (48, 48)


def load_fer2013(csv_path):
    # csv has 3 columns: emotion (0-6), pixels (2304 values separated by spaces), Usage
    print("Loading", csv_path)
    df = pd.read_csv(csv_path)

    X = np.stack([np.array(p.split(), dtype=np.float32) for p in df['pixels']])
    X = X.reshape(-1, 48, 48, 1) / 255.0
    y = to_categorical(df['emotion'].values, num_classes=len(EMOTIONS))

    if 'Usage' in df.columns:
        # use the official split
        train = (df['Usage'] == 'Training').values
        val = (df['Usage'] == 'PublicTest').values
        test = (df['Usage'] == 'PrivateTest').values
        X_train, y_train = X[train], y[train]
        X_val, y_val = X[val], y[val]
        X_test, y_test = X[test], y[test]
    else:
        # roughly 80/10/10
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y.argmax(1))
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.111, random_state=42, stratify=y_tmp.argmax(1))

    print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    counts = np.bincount(y_train.argmax(1), minlength=len(EMOTIONS))
    for emo, c in zip(EMOTIONS, counts):
        print(f"  {emo:<9}{c}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_weights(y_train):
    # FER2013 is very unbalanced (Disgust has ~400 train images, Happy ~7000)
    labels = y_train.argmax(1)
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))


def get_generators(X_train, y_train, X_val, y_val, batch_size=64):
    # light augmentation only - no vertical flips, faces are always upright
    aug = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    train_gen = aug.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen = ImageDataGenerator().flow(X_val, y_val, batch_size=batch_size, shuffle=False)
    return train_gen, val_gen


def preprocess_face(face_gray):
    # grayscale face crop -> (1, 48, 48, 1) float array, same format as training data
    face = cv2.resize(face_gray, IMG_SIZE).astype(np.float32) / 255.0
    return face.reshape(1, 48, 48, 1)
