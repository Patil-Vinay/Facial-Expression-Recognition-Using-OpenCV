# Facial Expression Recognition Using OpenCV

Detects faces in a webcam feed, video or image and classifies the expression on each one into
7 emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

It works in two stages:

1. **Face detection** with OpenCV's Haar cascade (`haarcascade_frontalface_default.xml`, ships with opencv-python so nothing extra to download)
2. **Emotion classification** with a CNN I trained from scratch on the FER2013 dataset

## Project structure

```
model/cnn.py           CNN architecture
utils/data_utils.py    dataset loading, augmentation, face preprocessing
train.py               trains the model and saves results to output/
detect.py              runs detection on webcam / video / image
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download `fer2013.csv` from [Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013) and put it in a `data/` folder.
The file is about 300 MB so it isn't included in this repo.

## Training

```bash
python train.py --csv data/fer2013.csv
```

Optional arguments: `--epochs` (default 100), `--batch` (default 64), `--lr` (default 0.001).
Early stopping usually ends training well before 100 epochs. On a GPU it takes around 15-20 minutes, on CPU expect 1-2 hours.

After training, `output/` contains:

- `emotion_cnn.h5` - best model (by validation accuracy)
- `training_curves.png` - accuracy and loss per epoch
- `confusion_matrix.png` - per-class results on the test set
- `training_log.csv`

## Running the detector

```bash
python detect.py                      # webcam
python detect.py --source video.mp4   # video file
python detect.py --image photo.jpg    # single image, result is saved as result_<name>
```

Press `q` to quit and `s` to save a screenshot.

## How it works

**Dataset.** FER2013 has 35,887 grayscale face images of 48x48 pixels. I used the official split
(28,709 train / 3,589 validation / 3,589 test). The classes are very unbalanced (Disgust has only ~400
training images while Happy has ~7,000), so the loss is weighted by class.

**Model.** A VGG-style network with about 2.5M parameters:

| Layer | Details |
|---|---|
| Conv block 1 | 2 x Conv 3x3, 32 filters, BatchNorm, ReLU, MaxPool, Dropout 0.25 |
| Conv block 2 | same with 64 filters |
| Conv block 3 | same with 128 filters |
| Conv block 4 | same with 256 filters |
| Dense | 512 then 256 units, BatchNorm, Dropout 0.5 |
| Output | 7 units, softmax |

L2 regularization is used on all conv layers. Training images are augmented with horizontal flips, small
rotations (up to 10 degrees), shifts and zoom. Adam optimizer with the learning rate halved whenever validation loss stops improving.

**Detection.** Each frame is converted to grayscale and histogram-equalized for the Haar cascade. Every
detected face is cropped with a small margin, resized to 48x48 and passed to the CNN. When there is only
one face in the frame, predictions are averaged over the last 5 frames so the label doesn't jump around.

## Notes

- FER2013 is a hard dataset, the labels are noisy and human accuracy on it is only around 65%.
- Haar cascades only find fairly frontal faces, so faces turned to the side are often missed.
- On FER2013 Fear and Sad are usually the most confused classes, while Happy and Surprise are the easiest.
