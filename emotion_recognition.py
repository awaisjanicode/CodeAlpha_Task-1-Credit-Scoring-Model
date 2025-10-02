#!/usr/bin/env python3
"""
Emotion Recognition from Speech
Save as: emotion_recognition.py

Usage examples:
  python emotion_recognition.py --build
  python emotion_recognition.py --train --epochs 20
  python emotion_recognition.py --predict path/to/file.wav

Requirements:
  pip install numpy librosa soundfile tensorflow scikit-learn tqdm
"""

import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
import math

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Reshape, Conv2D, MaxPool2D, TimeDistributed, GlobalAveragePooling2D, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Config
DATA_PATH = "data"
SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MFCC = 40
HOP_LENGTH = 512
FEATURE_CACHE = "feature_cache.npy"
LABEL_CACHE = "labels.npy"
CACHE_FEATURES = True
MODEL_SAVE = "emotion_recognition_model.h5"
TARGET_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]

def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    target_length = int(sr * duration)
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        pad_len = target_length - len(y)
        y = np.pad(y, (0, pad_len), 'constant')
    return y

def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=2048):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    # normalize per-coefficient
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)
    return mfcc.T  # (time_steps, n_mfcc)

def label_from_ravdess(filename):
    ravdess_map = {
        "01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fear","07":"disgust","08":"surprise"
    }
    parts = os.path.basename(filename).split('-')
    if len(parts) > 2:
        return ravdess_map.get(parts[2], None)
    return None

def label_from_tess(filename):
    name = os.path.basename(filename).lower()
    for e in TARGET_EMOTIONS:
        if e in name:
            return e
    return None

def infer_label(filepath):
    fname = os.path.basename(filepath).lower()
    lab = label_from_ravdess(fname)
    if lab: return lab
    lab = label_from_tess(fname)
    if lab: return lab
    parent = os.path.basename(os.path.dirname(filepath)).lower()
    for e in TARGET_EMOTIONS:
        if e in parent:
            return e
    return None

def gather_audio_files(data_root):
    audio_files = []
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith(('.wav', '.flac', '.mp3', '.aiff', '.aif')):
                audio_files.append(os.path.join(root, f))
    return audio_files

def build_dataset(data_root=DATA_PATH, cache=CACHE_FEATURES):
    if cache and os.path.exists(FEATURE_CACHE) and os.path.exists(LABEL_CACHE):
        print("Loading cached features...")
        X = np.load(FEATURE_CACHE, allow_pickle=True)
        y = np.load(LABEL_CACHE, allow_pickle=True)
        return X, y
    files = gather_audio_files(data_root)
    features = []
    labels = []
    print(f"Found {len(files)} audio files — extracting features...")
    for fp in tqdm(files):
        label = infer_label(fp)
        if label is None or label not in TARGET_EMOTIONS:
            continue
        try:
            y_audio = load_audio(fp)
            mf = extract_mfcc(y_audio)
            features.append(mf)
            labels.append(TARGET_EMOTIONS.index(label))
        except Exception as e:
            print(f"Error processing {fp}: {e}")
    if len(features) == 0:
        raise RuntimeError("No features extracted. Check your dataset and label mapping.")
    fixed_time = int(np.ceil((SAMPLE_RATE * DURATION) / HOP_LENGTH))
    X = np.zeros((len(features), fixed_time, N_MFCC), dtype=np.float32)
    for i, f in enumerate(features):
        t = min(f.shape[0], fixed_time)
        X[i, :t, :] = f[:t, :]
    y = np.array(labels, dtype=np.int32)
    if cache:
        np.save(FEATURE_CACHE, X)
        np.save(LABEL_CACHE, y)
    return X, y

def augment_shift_noise(x):
    max_shift = int(0.1 * x.shape[0])
    shift = np.random.randint(-max_shift, max_shift+1)
    if shift > 0:
        x = np.pad(x, ((shift,0),(0,0)), mode='constant')[:x.shape[0],:]
    elif shift < 0:
        x = np.pad(x, ((0,-shift),(0,0)), mode='constant')[-shift:,:]
    noise = np.random.normal(0, 0.005, size=x.shape)
    return x + noise

def prepare_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=random_state)
    y_train_cat = to_categorical(y_train, num_classes=len(TARGET_EMOTIONS))
    y_val_cat = to_categorical(y_val, num_classes=len(TARGET_EMOTIONS))
    y_test_cat = to_categorical(y_test, num_classes=len(TARGET_EMOTIONS))
    return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat)

def build_model(time_steps, n_mfcc=N_MFCC):
    inp = Input(shape=(time_steps, n_mfcc), name='mfcc_input')
    x = Reshape((time_steps, n_mfcc, 1))(inp)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.4)(x)
    out = Dense(len(TARGET_EMOTIONS), activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y, model_path=MODEL_SAVE, epochs=30, batch_size=32):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(X, y)
    model = build_model(X.shape[1], X.shape[2])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    def generator(Xarr, yarr, batch_size):
        n = Xarr.shape[0]
        idx = np.arange(n)
        while True:
            np.random.shuffle(idx)
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                Xb = Xarr[batch_idx].copy()
                for j in range(Xb.shape[0]):
                    if np.random.rand() < 0.4:
                        Xb[j] = augment_shift_noise(Xb[j])
                yield Xb, yarr[batch_idx]
    steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)
    history = model.fit(generator(X_train, y_train, batch_size),
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=callbacks,
                        verbose=2)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    return model, history

def predict_file(model, filepath):
    y_audio = load_audio(filepath)
    mf = extract_mfcc(y_audio)
    fixed_time = int(np.ceil((SAMPLE_RATE * DURATION) / HOP_LENGTH))
    X = np.zeros((1, fixed_time, N_MFCC), dtype=np.float32)
    t = min(mf.shape[0], fixed_time)
    X[0, :t, :] = mf[:t, :]
    pred = model.predict(X)
    idx = int(np.argmax(pred, axis=1)[0])
    return TARGET_EMOTIONS[idx], pred[0]

def main():
    parser = argparse.ArgumentParser(description="Emotion Recognition from Speech")
    parser.add_argument('--build', action='store_true', help='Build feature cache from data/')
    parser.add_argument('--train', action='store_true', help='Train model (requires built dataset or cached features)')
    parser.add_argument('--predict', type=str, default=None, help='Path to a WAV file to predict')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-cache', action='store_true', help='Disable caching extracted features')
    args = parser.parse_args()

    global CACHE_FEATURES
    if args.no_cache:
        CACHE_FEATURES = False

    if args.build:
        print("Building dataset and caching features...")
        X, y = build_dataset(DATA_PATH, cache=CACHE_FEATURES)
        print("Done. X shape:", X.shape, "y shape:", y.shape)
    if args.train:
        print("Loading/Building dataset for training...")
        X, y = build_dataset(DATA_PATH, cache=CACHE_FEATURES)
        print("Training model...")
        model, history = train_model(X, y, model_path=MODEL_SAVE, epochs=args.epochs, batch_size=args.batch_size)
        print("Model saved to", MODEL_SAVE)
    if args.predict:
        if not os.path.exists(args.predict):
            print("File not found:", args.predict)
            return
        if os.path.exists(MODEL_SAVE):
            model = load_model(MODEL_SAVE)
            emotion, scores = predict_file(model, args.predict)
            print(f"Predicted emotion: {emotion}")
            print("Class scores:", {TARGET_EMOTIONS[i]: float(scores[i]) for i in range(len(TARGET_EMOTIONS))})
        else:
            print("Trained model not found. Train model first with --train")

if __name__ == "__main__":
    main()
