import os, pdb
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/3ch/'

def setup():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]
    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.savefig(PNG_PATH + "some_digit_plot.png", dpi=300)
    print(y[36000])
    
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


if __name__ == '__main__':
    setup()