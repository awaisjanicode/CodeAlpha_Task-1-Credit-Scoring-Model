#!/usr/bin/env python3
"""
Handwritten Character Recognition (MNIST)
Save as: handwritten_recognition.py

Usage:
  python handwritten_recognition.py --train
  python handwritten_recognition.py --predict --index 0

Requirements:
  pip install numpy tensorflow
"""
import argparse
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

MODEL_SAVE = "mnist_cnn.h5"
BATCH_SIZE = 128
EPOCHS = 6

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        BatchNormalization(),
        MaxPool2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPool2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)
    y_train = to_categorical(y_train, 10)
    return x_train, y_train, x_test, y_test

def train_and_save(epochs=EPOCHS, batch_size=BATCH_SIZE, model_path=MODEL_SAVE):
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    model.summary()
    model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=2)
    model.save(model_path)
    print("Saved model to", model_path)

def evaluate_and_predict(index=0, model_path=MODEL_SAVE):
    x_train, y_train, x_test, y_test = load_data()
    if not os.path.exists(model_path):
        print("Model not found. Train first.")
        return
    model = load_model(model_path)
    loss, acc = model.evaluate(x_test, to_categorical(y_test,10), verbose=2)
    print(f"Test accuracy: {acc:.4f}")
    preds = model.predict(x_test[index:index+1])
    print("Predicted class:", int(np.argmax(preds, axis=1)[0]))
    print("True class:", int(y_test[index]))

def main():
    parser = argparse.ArgumentParser(description="Handwritten Character Recognition (MNIST)")
    parser.add_argument('--train', action='store_true', help='Train model on MNIST')
    parser.add_argument('--predict', action='store_true', help='Predict and evaluate using saved model')
    parser.add_argument('--index', type=int, default=0, help='Index of test sample to predict')
    parser.add_argument('--model-path', type=str, default=MODEL_SAVE)
    args = parser.parse_args()

    if args.train:
        train_and_save(model_path=args.model_path)
    elif args.predict:
        evaluate_and_predict(index=args.index, model_path=args.model_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
