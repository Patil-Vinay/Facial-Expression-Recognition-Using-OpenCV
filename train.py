# Train the emotion CNN on FER2013.
#
#   python train.py --csv data/fer2013.csv
#   python train.py --csv data/fer2013.csv --epochs 80 --batch 32
#
# Saves the best model, training curves, confusion matrix and a csv log to output/

import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from model.cnn import build_cnn, EMOTIONS
from utils.data_utils import load_fer2013, get_class_weights, get_generators

OUT_DIR = 'output'


def train(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_fer2013(args.csv)
    class_weights = get_class_weights(y_train)
    train_gen, val_gen = get_generators(X_train, y_train, X_val, y_val, args.batch)

    model = build_cnn()
    model.compile(optimizer=Adam(learning_rate=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    model_path = os.path.join(OUT_DIR, 'emotion_cnn.h5')
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(OUT_DIR, 'training_log.csv')),
    ]

    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        class_weight=class_weights)

    loss, acc = model.evaluate(X_test, y_test, batch_size=args.batch, verbose=0)
    print(f"\nTest accuracy: {acc * 100:.2f}%   test loss: {loss:.4f}")

    plot_history(history)
    plot_confusion_matrix(model, X_test, y_test, args.batch)
    print("Best model saved to", model_path)


def plot_history(history):
    h = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(h['accuracy'], label='train')
    ax1.plot(h['val_accuracy'], label='val')
    ax1.set_title('Accuracy')

    ax2.plot(h['loss'], label='train')
    ax2.plot(h['val_loss'], label='val')
    ax2.set_title('Loss')

    for ax in (ax1, ax2):
        ax.set_xlabel('epoch')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=120)
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, batch_size):
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0).argmax(1)
    y_true = y_test.argmax(1)

    print(classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion matrix (test set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'), dpi=120)
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='path to fer2013.csv')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    train(ap.parse_args())
