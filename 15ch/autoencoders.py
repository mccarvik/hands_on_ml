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


def cache_frozen_layer():
    pass


def one_encoder_single_graphs():
    mnist = input_data.read_data_sets("/tmp/data/")
    reset_graph()
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 150  # codings
    n_hidden3 = n_hidden1
    n_outputs = n_inputs
    learning_rate = 0.01
    l2_reg = 0.0001
    
    activation = tf.nn.elu
    regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    initializer = tf.contrib.layers.variance_scaling_initializer()
    
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    
    weights1_init = initializer([n_inputs, n_hidden1])
    weights2_init = initializer([n_hidden1, n_hidden2])
    weights3_init = initializer([n_hidden2, n_hidden3])
    weights4_init = initializer([n_hidden3, n_outputs])
    
    weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
    weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
    weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
    weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")
    
    biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
    biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
    biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
    biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")
    
    hidden1 = activation(tf.matmul(X, weights1) + biases1)
    hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
    hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
    outputs = tf.matmul(hidden3, weights4) + biases4
    
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    with tf.name_scope("phase1"):
        phase1_outputs = tf.matmul(hidden1, weights4) + biases4  # bypass hidden2 and hidden3
        phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs - X))
        phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
        phase1_loss = phase1_reconstruction_loss + phase1_reg_loss
        phase1_training_op = optimizer.minimize(phase1_loss)
    
    with tf.name_scope("phase2"):
        phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
        phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
        phase2_loss = phase2_reconstruction_loss + phase2_reg_loss
        train_vars = [weights2, biases2, weights3, biases3]
        phase2_training_op = optimizer.minimize(phase2_loss, var_list=train_vars) # freeze hidden1

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    training_ops = [phase1_training_op, phase2_training_op]
    reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
    n_epochs = [4, 4]
    batch_sizes = [150, 150]
    with tf.Session() as sess:
        init.run()
        for phase in range(2):
            print("Training phase #{}".format(phase + 1))
            for epoch in range(n_epochs[phase]):
                n_batches = mnist.train.num_examples // batch_sizes[phase]
                for iteration in range(n_batches):
                    print("\r{}%".format(100 * iteration // n_batches), end="")
                    sys.stdout.flush()
                    X_batch, y_batch = mnist.train.next_batch(batch_sizes[phase])
                    sess.run(training_ops[phase], feed_dict={X: X_batch})
                loss_train = reconstruction_losses[phase].eval(feed_dict={X: X_batch})
                print("\r{}".format(epoch), "Train MSE:", loss_train)
                saver.save(sess, "./my_model_one_at_a_time.ckpt")
        loss_test = reconstruction_loss.eval(feed_dict={X: mnist.test.images})
        print("Test MSE:", loss_test)


def one_encoder_mult_graphs():
    mnist = input_data.read_data_sets("/tmp/data/")
    reset_graph()
    
    hidden_output, W1, b1, W4, b4 = train_autoencoder(mnist.train.images, n_neurons=300, n_epochs=4, batch_size=150, output_activation=None)
    _, W2, b2, W3, b3 = train_autoencoder(hidden_output, n_neurons=150, n_epochs=4, batch_size=150)
    reset_graph()
    n_inputs = 28*28
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden1 = tf.nn.elu(tf.matmul(X, W1) + b1)
    hidden2 = tf.nn.elu(tf.matmul(hidden1, W2) + b2)
    hidden3 = tf.nn.elu(tf.matmul(hidden2, W3) + b3)
    outputs = tf.matmul(hidden3, W4) + b4
    show_reconstructed_digits(X, outputs, None)
    plt.savefig(PNG_PATH + "one_encoder_mult_graphs", dpi=300)
    plt.close()


def tying_weights():
    # tie the weights of the encoder and the decoder
    mnist = input_data.read_data_sets("/tmp/data/")
    reset_graph()
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 150  # codings
    n_hidden3 = n_hidden1
    n_outputs = n_inputs
    
    learning_rate = 0.01
    l2_reg = 0.0005
    activation = tf.nn.elu
    regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    initializer = tf.contrib.layers.variance_scaling_initializer()
    
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    
    weights1_init = initializer([n_inputs, n_hidden1])
    weights2_init = initializer([n_hidden1, n_hidden2])
    
    weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
    weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
    weights3 = tf.transpose(weights2, name="weights3")  # tied weights
    weights4 = tf.transpose(weights1, name="weights4")  # tied weights
    
    biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
    biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
    biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
    biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")
    
    hidden1 = activation(tf.matmul(X, weights1) + biases1)
    hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
    hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
    outputs = tf.matmul(hidden3, weights4) + biases4
    
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    reg_loss = regularizer(weights1) + regularizer(weights2)
    loss = reconstruction_loss + reg_loss
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    n_epochs = 4
    batch_size = 150
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
            saver.save(sess, "./my_model_tying_weights.ckpt")
    pdb.set_trace()
    show_reconstructed_digits(X, outputs, "./my_model_tying_weights.ckpt")
    plt.savefig(PNG_PATH + "tying_weights", dpi=300)
    plt.close()


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
    
    # Tie Weights
    # tie the weights of the encoder and the decoder
    mnist = input_data.read_data_sets("/tmp/data/")
    reset_graph()
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 150  # codings
    n_hidden3 = n_hidden1
    n_outputs = n_inputs
    
    learning_rate = 0.01
    l2_reg = 0.0005
    activation = tf.nn.elu
    regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    initializer = tf.contrib.layers.variance_scaling_initializer()
    
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    
    weights1_init = initializer([n_inputs, n_hidden1])
    weights2_init = initializer([n_hidden1, n_hidden2])
    
    weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
    weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
    weights3 = tf.transpose(weights2, name="weights3")  # tied weights
    weights4 = tf.transpose(weights1, name="weights4")  # tied weights
    
    biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
    biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
    biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
    biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")
    
    hidden1 = activation(tf.matmul(X, weights1) + biases1)
    hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
    hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
    outputs = tf.matmul(hidden3, weights4) + biases4
    
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    reg_loss = regularizer(weights1) + regularizer(weights2)
    loss = reconstruction_loss + reg_loss
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    n_epochs = 4
    batch_size = 150
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
            saver.save(sess, "./my_model_tying_weights.ckpt")
    pdb.set_trace()
    show_reconstructed_digits(X, outputs, "./my_model_tying_weights.ckpt")
    plt.savefig(PNG_PATH + "tying_weights", dpi=300)
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


def train_autoencoder(X_train, n_neurons, n_epochs, batch_size, learning_rate = 0.01, l2_reg = 0.0005, seed=42, hidden_activation=tf.nn.elu, output_activation=tf.nn.elu):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)
        n_inputs = X_train.shape[1]
        X = tf.placeholder(tf.float32, shape=[None, n_inputs])
        
        my_dense_layer = partial(tf.layers.dense,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        hidden = my_dense_layer(X, n_neurons, activation=hidden_activation, name="hidden")
        outputs = my_dense_layer(hidden, n_inputs, activation=output_activation, name="outputs")
        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = len(X_train) // batch_size
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                indices = rnd.permutation(len(X_train))[:batch_size]
                X_batch = X_train[indices]
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
        params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        hidden_val = hidden.eval(feed_dict={X: X_train})
        return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["outputs/kernel:0"], params["outputs/bias:0"]


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
    # stacked_autoencoders()
    tying_weights()
    # one_encoder_mult_graphs()
    # one_encoder_single_graphs()