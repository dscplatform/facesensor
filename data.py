import numpy as np
from PIL import Image
import keras
import matplotlib.pyplot as plt

def build_dataset():
    subjects = 40
    unique = 10
    X = np.zeros((400, 1, 112, 92), dtype=np.uint8)
    y = np.zeros(400, dtype=np.uint8)
    i = 0
    for u in range(1, unique + 1):
        for s in range(1, subjects + 1):
            im = Image.open(("data/s%d/%d.pgm")%(s, u))
            X[i] = [np.asarray(im)]
            y[i] = s - 1
            i+=1
    X = X.astype("float32")
    X /= 255
    y = keras.utils.to_categorical(y, subjects)
    return (X, y)

def visualize(data):
    plt.title("Data")
    plt.imshow(data, cmap="gray")
    plt.show()
    #im = Image.open("data/s1/1.pgm")
    #im.show()
    #arr = np.asarray(im)
