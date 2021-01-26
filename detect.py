# Real-time facial expression recognition.
# Faces are found with OpenCV's Haar cascade, then each face is classified by the trained CNN.
#
#   python detect.py                      # webcam
#   python detect.py --source video.mp4   # video file
#   python detect.py --image photo.jpg    # single image
#
# Keys: q = quit, s = save screenshot

import os
import time
import argparse
from collections import deque

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

from model.cnn import EMOTIONS
from utils.data_utils import preprocess_face

# BGR
COLORS = {
    'Angry': (0, 0, 230),
    'Disgust': (0, 140, 0),
    'Fear': (160, 32, 240),
    'Happy': (0, 215, 255),
    'Sad': (255, 144, 30),
    'Surprise': (0, 165, 255),
    'Neutral': (180, 180, 180),
}

CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


class EmotionDetector:
    def __init__(self, model_path, smooth=5):
        if not os.path.exists(model_path):
            raise SystemExit(f"Model not found: {model_path}\n"
                             "Train it first: python train.py --csv data/fer2013.csv")

        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(CASCADE)
        if self.face_cascade.empty():
            raise IOError("Could not load " + CASCADE)

        # average the last few predictions so the label doesn't flicker every frame
        self.history = deque(maxlen=smooth)

    def detect_faces(self, gray):
        # equalizing helps the cascade in bad lighting
        return self.face_cascade.detectMultiScale(cv2.equalizeHist(gray),
                                                  scaleFactor=1.1,
                                                  minNeighbors=5,
                                                  minSize=(48, 48))

    def predict(self, face_gray, smooth):
        blob = preprocess_face(face_gray)
        probs = self.model(blob, training=False).numpy()[0]   # much faster than model.predict for 1 image

        if smooth:
            self.history.append(probs)
            probs = np.mean(self.history, axis=0)

        i = int(np.argmax(probs))
        return EMOTIONS[i], float(probs[i]), probs

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        if len(faces) == 0:
            self.history.clear()
            h, w = frame.shape[:2]
            cv2.putText(frame, "No face detected", (w // 2 - 145, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (60, 60, 255), 2, cv2.LINE_AA)
            return frame

        # smoothing only makes sense when we're tracking one person,
        # with several faces the buffer would mix their predictions
        smooth = len(faces) == 1
        if not smooth:
            self.history.clear()

        for (x, y, w, h) in faces:
            # small margin around the face, FER2013 crops include a bit of hair/chin
            pad = int(0.08 * min(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

            # crop from the non-equalized image, the model was trained on raw pixels
            emotion, conf, probs = self.predict(gray[y1:y2, x1:x2], smooth)

            draw_box(frame, x1, y1, x2, y2, emotion, conf)
            draw_bars(frame, x2 + 12, y1, probs, emotion)

        return frame


def draw_box(frame, x1, y1, x2, y2, emotion, conf):
    color = COLORS[emotion]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{emotion} {conf * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_bars(frame, x, y, probs, top):
    # one small bar per emotion next to the face
    bar_w, bar_h, gap = 130, 15, 5
    for i, (emo, p) in enumerate(zip(EMOTIONS, probs)):
        by = y + i * (bar_h + gap)
        color = COLORS[top] if emo == top else (60, 60, 60)
        cv2.rectangle(frame, (x, by), (x + bar_w, by + bar_h), (35, 35, 35), -1)
        cv2.rectangle(frame, (x, by), (x + int(bar_w * p), by + bar_h), color, -1)
        cv2.putText(frame, f"{emo[:4]} {p * 100:.0f}%", (x + bar_w + 6, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


def run_video(detector, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    prev = time.time()
    shots = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = detector.process_frame(frame)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-5)
        prev = now
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 220, 50), 2, cv2.LINE_AA)

        cv2.imshow("Emotion detector (q to quit, s to save)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            name = f"screenshot_{shots:03d}.jpg"
            cv2.imwrite(name, frame)
            print("saved", name)
            shots += 1

    cap.release()
    cv2.destroyAllWindows()


def run_image(detector, path):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(path)

    result = detector.process_frame(frame)
    out = "result_" + os.path.basename(path)
    cv2.imwrite(out, result)
    print("saved", out)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='output/emotion_cnn.h5')
    ap.add_argument('--source', default='0', help='webcam index or path to a video')
    ap.add_argument('--image', help='run on a single image instead')
    ap.add_argument('--smooth', type=int, default=5, help='number of frames to average over')
    args = ap.parse_args()

    detector = EmotionDetector(args.model, args.smooth)
    if args.image:
        run_image(detector, args.image)
    else:
        run_video(detector, int(args.source) if args.source.isdigit() else args.source)
