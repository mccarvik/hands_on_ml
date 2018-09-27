import os, pdb, time, sys, time
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap


PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/10ch/'


def percept():
    # simple perceptron example
    iris = load_iris()
    X = iris.data[:, (2, 3)]  # petal length, petal width
    y = (iris.target == 0).astype(np.int)

    per_clf = Perceptron(max_iter=100, random_state=42)
    per_clf.fit(X, y)

    y_pred = per_clf.predict([[2, 0.5]])
    print(y_pred)
    
    # Linearly separable graphical example using Perceptron
    a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
    b = -per_clf.intercept_ / per_clf.coef_[0][1]
    
    axes = [0, 5, 0, 2]
    
    x0, x1 = np.meshgrid(
            np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
            np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
        )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = per_clf.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    
    plt.figure(figsize=(10, 4))
    plt.plot(X[y==0, 0], X[y==0, 1], "bs", label="Not Iris-Setosa")
    plt.plot(X[y==1, 0], X[y==1, 1], "yo", label="Iris-Setosa")
    plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)
    custom_cmap = ListedColormap(['#9898ff', '#fafab0'])
    
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.axis(axes)
    plt.savefig(PNG_PATH + "perceptron_iris_plot", dpi=300)
    plt.close()


def act_functions():
    z = np.linspace(-5, 5, 200)
    plt.figure(figsize=(11,4))
    
    # showing the different shapes for each activation function
    plt.subplot(121)
    plt.plot(z, np.sign(z), "r-", linewidth=2, label="Step")
    plt.plot(z, logit(z), "g--", linewidth=2, label="Logit")
    plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)
    plt.legend(loc="center right", fontsize=14)
    plt.title("Activation functions", fontsize=14)
    plt.axis([-5, 5, -1.2, 1.2])
    
    plt.subplot(122)
    plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Step")
    plt.plot(0, 0, "ro", markersize=5)
    plt.plot(0, 0, "rx", markersize=10)
    plt.plot(z, derivative(logit, z), "g--", linewidth=2, label="Logit")
    plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)
    plt.legend(loc="center right", fontsize=14)
    plt.title("Derivatives", fontsize=14)
    plt.axis([-5, 5, -0.2, 1.2])
    
    plt.savefig(PNG_PATH + "activation_functions_plot", dpi=300)
    plt.close()
    
    
    
def logit(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)    


if __name__ == '__main__':
    # percept()
    act_functions()