import os, pdb, time, sys, time
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy.random as rnd
from functools import partial
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/15ch/'


def stacked_autoencoders():
    mnist = input_data.read_data_sets("/tmp/data/")
    reset_graph()
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 150  # codings
    n_hidden3 = n_hidden1
    n_outputs = n_inputs
    learning_rate = 0.01
    l2_reg = 0.0001
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    
    he_init = tf.contrib.layers.variance_scaling_initializer() # He initialization
    #Equivalent to:
    #he_init = lambda shape, dtype=tf.float32: tf.truncated_normal(shape, 0., stddev=np.sqrt(2/shape[0]))
    l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    my_dense_layer = partial(tf.layers.dense,
                             activation=tf.nn.elu,
                             kernel_initializer=he_init,
                             kernel_regularizer=l2_regularizer)
    
    hidden1 = my_dense_layer(X, n_hidden1)
    hidden2 = my_dense_layer(hidden1, n_hidden2)
    hidden3 = my_dense_layer(hidden2, n_hidden3)
    outputs = my_dense_layer(hidden3, n_outputs, activation=None)
    
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([reconstruction_loss] + reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() # not shown in the book

    n_epochs = 5
    batch_size = 150
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="") # not shown in the book
                sys.stdout.flush()                                          # not shown
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})   # not shown
            print("\r{}".format(epoch), "Train MSE:", loss_train)           # not shown
        saver.save(sess, "./my_model_all_layers.ckpt")
    show_reconstructed_digits(X, outputs, saver, "./my_model_all_layers.ckpt")
    plt.savefig(PNG_PATH + "reconstruction_plot", dpi=300)
    plt.close()


def pca_with_linear_autoencoder():
    rnd.seed(4)
    m = 200
    w1, w2 = 0.1, 0.3
    noise = 0.1
    angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
    data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
    data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data[:100])
    X_test = scaler.transform(data[100:])
    
    reset_graph()
    n_inputs = 3
    n_hidden = 2  # codings
    n_outputs = n_inputs
    
    learning_rate = 0.01
    
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = tf.layers.dense(X, n_hidden)
    outputs = tf.layers.dense(hidden, n_outputs)
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(reconstruction_loss)
    init = tf.global_variables_initializer()
    n_iterations = 1000
    codings = hidden
    
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            training_op.run(feed_dict={X: X_train})
        codings_val = codings.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(4,3))
    plt.plot(codings_val[:,0], codings_val[:, 1], "b.")
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.savefig(PNG_PATH + "linear_autoencoder_pca_plot", dpi=300)
    plt.close()
    

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def show_reconstructed_digits(X, outputs, saver, model_path = None, n_test_digits = 2):
    mnist = input_data.read_data_sets("/tmp/data/")
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # make the minimum == 0, so the padding looks white
    w,h = images.shape[1:]
    image = np.zeros(((w+pad)*n_rows+pad, (h+pad)*n_cols+pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h),(x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y*n_cols+x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")


if __name__ == '__main__':
    # pca_with_linear_autoencoder()
    stacked_autoencoders()