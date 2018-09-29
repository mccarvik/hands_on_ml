import os, pdb, time, sys, time
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/11ch/'


def saturation():
    # at 1 and 0 the function saturates, little left for the lower layers to learn
    z = np.linspace(-5, 5, 200)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [1, 1], 'k--')
    plt.plot([0, 0], [-0.2, 1.2], 'k-')
    plt.plot([-5, 5], [-3/4, 7/4], 'g--')
    plt.plot(z, logit(z), "b-", linewidth=2)
    props = dict(facecolor='black', shrink=0.1)
    plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
    plt.grid(True)
    plt.title("Sigmoid activation function", fontsize=14)
    plt.axis([-5, 5, -0.2, 1.2])
    plt.savefig(PNG_PATH + "sigmoid_saturation_plot", dpi=300)
    plt.close()
    

def xavier_he_initialization():
    reset_graph()
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    he_init = tf.variance_scaling_initializer()
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, kernel_initializer=he_init, name="hidden1")


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def logit(z):
    return 1 / (1 + np.exp(-z))

if __name__ == '__main__':
    # saturation()
    xavier_he_initialization()