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


def elu_act_mnist():
    reset_graph()
    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu_new, name="hidden1")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu, name="hidden2")
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    pdb.set_trace()
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:50], X_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]

    n_epochs = 10
    batch_size = 50

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if epoch % 5 == 0:
                acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
                print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)
    
        save_path = saver.save(sess, "./my_model_final.ckpt")


def selu_graph():
    z = np.linspace(-5, 5, 200)
    plt.plot(z, selu(z), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [-1.758, -1.758], 'k--')
    plt.plot([0, 0], [-2.2, 3.2], 'k-')
    plt.grid(True)
    plt.title(r"SELU activation function", fontsize=14)
    plt.axis([-5, 5, -2.2, 3.2])
    plt.savefig(PNG_PATH + "selu_plot", dpi=300)
    plt.close()


def exponential_linear_unit_graph():
    pdb.set_trace()
    plt.plot(z, elu(z), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [-1, -1], 'k--')
    plt.plot([0, 0], [-2.2, 3.2], 'k-')
    plt.grid(True)
    plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
    plt.axis([-5, 5, -2.2, 3.2])
    plt.savefig(PNG_PATH + "elu_plot", dpi=300)
    plt.close()
    
    # Using ELU with tensorflow
    # reset_graph()
    # X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    # hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name="hidden1")


def leaky_relu_graph():
    z = np.linspace(-5, 5, 200)
    plt.plot(z, leaky_relu(z, 0.05), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([0, 0], [-0.5, 4.2], 'k-')
    plt.grid(True)
    props = dict(facecolor='black', shrink=0.1)
    plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
    plt.title("Leaky ReLU activation function", fontsize=14)
    plt.axis([-5, 5, -0.5, 4.2])
    plt.savefig(PNG_PATH + "leaky_relu_plot", dpi=300)
    plt.close()


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

def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)

def leaky_relu_new(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

def selu(z, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    # like elu but maintains variance eliminating a lot of the gradient dexcent vanishing/expoding issues
    return scale * elu(z, alpha)


if __name__ == '__main__':
    # saturation()
    # xavier_he_initialization()
    # leaky_relu_graph()
    exponential_linear_unit_graph()
    selu_graph()