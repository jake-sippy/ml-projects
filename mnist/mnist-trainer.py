import struct
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pickle

TRAIN_IMAGE_PATH="train-images-ubyte"
TRAIN_LABELS_PATH="train-labels-ubyte"
TEST_IMAGE_PATH="test-images-ubyte"
TEST_LABELS_PATH="test-labels-ubyte"


# Reading the ubyte files
with open(TRAIN_IMAGE_PATH, "rb") as f:
    magic_num, images, rows, cols = struct.unpack(">IIII", f.read(16))
    X_train = np.fromfile(f, dtype="uint8").reshape(images, rows*cols)

with open(TRAIN_LABELS_PATH, "rb") as f:
    magic_num, labels = struct.unpack(">II", f.read(8))
    y_train = np.fromfile(f, dtype="uint8")

with open(TEST_IMAGE_PATH, "rb") as f:
    magic_num, images, rows, cols = struct.unpack(">IIII", f.read(16))
    X_test = np.fromfile(f, dtype="uint8").reshape(images, rows*cols)

with open(TEST_LABELS_PATH, "rb") as f:
    magic_num, labels = struct.unpack(">II", f.read(8))
    y_test = np.fromfile(f, dtype="uint8")

print("Training")
clf = MLPClassifier()
clf.fit(X_train, y_train)
print("Predicting")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
