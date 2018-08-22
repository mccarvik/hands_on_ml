import os, pdb, time, sys, time
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
# from tensorflow_graph_in_jupyter import show_graph
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/9ch/'


def sharing_variables():
    reset_graph()
    n_features = 3
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    
    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
        indices = np.random.randint(m, size=batch_size)  # not shown
        X_batch = scaled_housing_data_plus_bias[indices] # not shown
        y_batch = housing.target.reshape(-1, 1)[indices] # not shown
        return X_batch, y_batch
        
    reset_graph()
    # shared variable threshold but have to pass in variables one by one this way
    def relu1(X, threshold):
        with tf.name_scope("relu1"):
            w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
            w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
            b = tf.Variable(0.0, name="bias")                           # not shown
            z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
            return tf.maximum(z, threshold, name="max")
    
    threshold = tf.Variable(0.0, name="threshold")
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = [relu1(X, threshold) for i in range(5)]
    output = tf.add_n(relus, name="output")

    reset_graph()
    # Set the shared variable as an attribute of the relu function on the first call
    def relu2(X):
        with tf.name_scope("relu2"):
            if not hasattr(relu2, "threshold"):
                relu2.threshold = tf.Variable(0.0, name="threshold")
            w_shape = int(X.get_shape()[1]), 1                          # not shown in the book
            w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
            b = tf.Variable(0.0, name="bias")                           # not shown
            z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
            return tf.maximum(z, relu2.threshold, name="max")
    
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = [relu2(X) for i in range(5)]
    output = tf.add_n(relus, name="output")

    reset_graph()
    # create variable if it does not already exist thru 'get_variable' function
    with tf.variable_scope("relu3"):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))

    with tf.variable_scope("relu3", reuse=True):
        threshold = tf.get_variable("threshold")

    with tf.variable_scope("relu3") as scope:
        scope.reuse_variables()
        threshold = tf.get_variable("threshold")

    reset_graph()
    # for this threshold has to be defined outside the relu function
    def relu3(X):
        with tf.variable_scope("relu3", reuse=True):
            threshold = tf.get_variable("threshold")
            w_shape = int(X.get_shape()[1]), 1                          # not shown
            w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
            b = tf.Variable(0.0, name="bias")                           # not shown
            z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
            return tf.maximum(z, threshold, name="max")
    
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    with tf.variable_scope("relu3"):
        threshold = tf.get_variable("threshold", shape=(),
                                    initializer=tf.constant_initializer(0.0))
    relus = [relu3(X) for relu_index in range(5)]
    output = tf.add_n(relus, name="output")

    file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
    file_writer.close()

    reset_graph()
    # here threshold is defined inside the relu function
    def relu4(X):
        with tf.variable_scope("relu4"):
            threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
            w_shape = (int(X.get_shape()[1]), 1)
            w = tf.Variable(tf.random_normal(w_shape), name="weights")
            b = tf.Variable(0.0, name="bias")
            z = tf.add(tf.matmul(X, w), b, name="z")
            return tf.maximum(z, threshold, name="max")
    
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    with tf.variable_scope("", default_name="") as scope:
        first_relu = relu4(X)     # create the shared variable
        scope.reuse_variables()  # then reuse it
        relus = [first_relu] + [relu4(X) for i in range(4)]
    output = tf.add_n(relus, name="output")
    
    file_writer = tf.summary.FileWriter("logs/relu8", tf.get_default_graph())
    file_writer.close()

    reset_graph()
    def relu5(X):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
        w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, threshold, name="max")
    
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = []
    # creates the relus dynamically
    for relu_index in range(5):
        with tf.variable_scope("relu5", reuse=(relu_index >= 1)) as scope:
            relus.append(relu5(X))
    output = tf.add_n(relus, name="output")

    file_writer = tf.summary.FileWriter("logs/relu9", tf.get_default_graph())
    file_writer.close()


def name_scopes():
    # Can display metrics in Jupyter 
    reset_graph()
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    
    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
        indices = np.random.randint(m, size=batch_size)  # not shown
        X_batch = scaled_housing_data_plus_bias[indices] # not shown
        y_batch = housing.target.reshape(-1, 1)[indices] # not shown
        return X_batch, y_batch

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    
    # define error and mse within name scope 'loss'
    with tf.name_scope("loss") as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name="mse")
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    
    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        best_theta = theta.eval()

    file_writer.flush()
    file_writer.close()
    print("Best theta:")
    print(best_theta)
    # printing out name scopes
    print(error.op.name)
    print(mse.op.name)
    
    reset_graph()
    a1 = tf.Variable(0, name="a")      # name == "a"
    a2 = tf.Variable(0, name="a")      # name == "a_1"
    
    with tf.name_scope("param"):       # name == "param"
        a3 = tf.Variable(0, name="a")  # name == "param/a"
    
    with tf.name_scope("param"):       # name == "param_1"
        a4 = tf.Variable(0, name="a")  # name == "param_1/a"
    
    for node in (a1, a2, a3, a4):
        print(node.op.name)
    
    
def modularity():
    # Can display metrics in Jupyter 
    reset_graph()
    
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    
    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
        indices = np.random.randint(m, size=batch_size)  # not shown
        X_batch = scaled_housing_data_plus_bias[indices] # not shown
        y_batch = housing.target.reshape(-1, 1)[indices] # not shown
        return X_batch, y_batch
    
    
    # UGLY CODE that tensorflow replaces
    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    
    w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
    w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
    b1 = tf.Variable(0.0, name="bias1")
    b2 = tf.Variable(0.0, name="bias2")
    
    z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
    z2 = tf.add(tf.matmul(X, w2), b2, name="z2")
    
    relu1 = tf.maximum(z1, 0., name="relu1")
    relu2 = tf.maximum(z1, 0., name="relu2")  # Oops, cut&paste error! Did you spot it?
    
    output = tf.add(relu1, relu2, name="output")
    reset_graph()

    # function that will compute the sum of a list of tensors
    def relu(X):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, 0., name="relu")

    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = [relu(X) for i in range(5)]
    output = tf.add_n(relus, name="output")
    file_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())

    reset_graph()
    
    # Even better implementation using name scopes
    def relu2(X):
        with tf.name_scope("relu2"):
            w_shape = (int(X.get_shape()[1]), 1)                          # not shown in the book
            w = tf.Variable(tf.random_normal(w_shape), name="weights")    # not shown
            b = tf.Variable(0.0, name="bias")                             # not shown
            z = tf.add(tf.matmul(X, w), b, name="z")                      # not shown
            return tf.maximum(z, 0., name="max")                          # not shown

    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = [relu2(X) for i in range(5)]
    output = tf.add_n(relus, name="output")

    file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
    file_writer.close()


def using_tensorboard():
    # Can display metrics in Jupyter 
    # show_graph(tf.get_default_graph())
    reset_graph()
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    
    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
        indices = np.random.randint(m, size=batch_size)  # not shown
        X_batch = scaled_housing_data_plus_bias[indices] # not shown
        y_batch = housing.target.reshape(-1, 1)[indices] # not shown
        return X_batch, y_batch

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))
    
    with tf.Session() as sess:                                                        # not shown in the book
        sess.run(init)                                                                # not shown
    
        for epoch in range(n_epochs):                                                 # not shown
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    
        best_theta = theta.eval()                                                     # not shown
    file_writer.close()
    print(best_theta)


def saving_restoring_model():
    reset_graph()
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    n_epochs = 1000                                                                     # not shown in the book
    learning_rate = 0.01                                                                  # not shown

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")            # not shown
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")            # not shown
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")                                      # not shown
    error = y_pred - y                                                                    # not shown
    mse = tf.reduce_mean(tf.square(error), name="mse")                                    # not shown
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            # not shown
    training_op = optimizer.minimize(mse)                                                 # not shown

    init = tf.global_variables_initializer()
    # create a Saver node
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
                # call the sav method during execution
                save_path = saver.save(sess, "./my_model.ckpt")
            sess.run(training_op)
    
        best_theta = theta.eval()
        save_path = saver.save(sess, "./my_model_final.ckpt")
        print(best_theta)
    
    with tf.Session() as sess:
        # Call restore to get model back from saved file
        saver.restore(sess, "./my_model_final.ckpt")
        best_theta_restored = theta.eval() # not shown in the book
    
    print(np.allclose(best_theta, best_theta_restored))

    # specify which variables get saved
    saver = tf.train.Saver({"weights": theta})

    reset_graph()
    # notice that we start with an empty graph.

    # by default structure of graph gets saved, can load structure like this
    saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")  # this loads the graph structure
    theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")  # this restores the graph's state
        best_theta_restored = theta.eval() # not shown in the book
    print(np.allclose(best_theta, best_theta_restored))


def feeding_data_to_algo():
    reset_graph()
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    # Placeholder nodes - usually just used to pass data
    A = tf.placeholder(tf.float32, shape=(None, 3))
    B = A + 5
    with tf.Session() as sess:
        B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
        B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
    print(B_val_1)
    print(B_val_2)
    
    # adjusting gradient descent with placeholders
    n_epochs = 1000
    learning_rate = 0.01
    reset_graph()
    
    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
        indices = np.random.randint(m, size=batch_size)  # not shown
        X_batch = scaled_housing_data_plus_bias[indices] # not shown
        y_batch = housing.target.reshape(-1, 1)[indices] # not shown
        return X_batch, y_batch

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)
    
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                # grab each batch
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    
        best_theta = theta.eval()
        print(best_theta)


def using_optimizers():
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01
    
    # raw data constants
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    # variables to calc
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
    
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)
        
        best_theta = theta.eval()
    
    print("Best theta:")
    print(best_theta)

    reset_graph()
    
    # Same as above just different optimizer
    n_epochs = 1000
    learning_rate = 0.01
    
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9)
    training_op = optimizer.minimize(mse)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
    
        for epoch in range(n_epochs):
            sess.run(training_op)
        
        best_theta = theta.eval()
    
    print("Best theta:")
    print(best_theta)


def using_auto_diff():
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01
    
    # raw data as constants
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    # variables to be calc'd using tf methods
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    # automatically and efficeintly compute the gradients
    gradients = tf.gradients(mse, [theta])[0]
    training_op = tf.assign(theta, theta - learning_rate * gradients)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
    
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)
        
        best_theta = theta.eval()

    print("Best theta:")
    print(best_theta)
    
    # Run the gradient calc on complicated function
    print(my_func(0.2, 0.3))
    reset_graph()

    a = tf.Variable(0.2, name="a")
    b = tf.Variable(0.3, name="b")
    z = tf.constant(0.0, name="z0")
    for i in range(100):
        z = a * tf.cos(z + i) + z * tf.sin(b - i)
    
    grads = tf.gradients(z, [a, b])
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        print(z.eval())
        print(sess.run(grads))
    

def my_func(a, b):
    z = 0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    return z


def grad_desc():
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    print(scaled_housing_data_plus_bias.mean(axis=0))
    print(scaled_housing_data_plus_bias.mean(axis=1))
    print(scaled_housing_data_plus_bias.mean())
    print(scaled_housing_data_plus_bias.shape)
    
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01
    
    # setting up computations to be put on graph
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    
    # random_uniform - creates a node in the graph that will produce random values
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    gradients = 2/m * tf.matmul(tf.transpose(X), error)
    # assign - function that will assign a new value to a variable, in this case batch grad desc
    training_op = tf.assign(theta, theta - learning_rate * gradients)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
    
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)
        
        best_theta = theta.eval()
    print(best_theta)


def lin_reg():
    reset_graph()
    # grab data
    housing = fetch_california_housing()
    m, n = housing.data.shape
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
    
    pdb.set_trace()
    # These dont perform the computations until the graph is run
    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
    
    with tf.Session() as sess:
        theta_value = theta.eval()
    print(theta_value)

    X = housing_data_plus_bias
    y = housing.target.reshape(-1, 1)
    theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    print(theta_numpy)

    lin_reg = LinearRegression()
    lin_reg.fit(housing.data, housing.target.reshape(-1, 1))
    print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])
    

def managing_graphs():
    reset_graph()

    # Any node created is automatically added to default graph
    x1 = tf.Variable(1)
    print(x1.graph is tf.get_default_graph())

    # creating a new graph and making it default while in the with block
    graph = tf.Graph()
    with graph.as_default():
        x2 = tf.Variable(2)
    
    print(x2.graph is graph)
    print(x2.graph is tf.get_default_graph())
    
    # when you evaluate a node, tensorflow automatically determines the set of nodes that it depends on
    w = tf.constant(3)
    x = w + 2
    y = x + 5
    z = x * 3
    
    # y and z evaluated twive
    with tf.Session() as sess:
        print(y.eval())  # 10
        print(z.eval())  # 15

    # evaluating y and z more efficiently
    with tf.Session() as sess:
        y_val, z_val = sess.run([y, z])
        print(y_val)  # 10
        print(z_val)  # 15


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def create_graph():
    reset_graph()

    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x*x*y + y + 2
    print(f)
    
    # creates a session, initializes the variables, and evaluates and then closes
    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print(result)
    sess.close()

    # eliminates the need to call sess.run() all the time, also automativally closes the session
    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()
    print(result)

    # prepare an init node for variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # actually initialize all the variables
        init.run()
        result = f.eval()
    print(result)

    # Interactive session --> automatically sets itself as the default session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    init.run()
    result = f.eval()
    print(result)
    sess.close()


if __name__ == '__main__':
    # create_graph()
    # managing_graphs()
    # lin_reg()
    # grad_desc()
    # using_auto_diff()
    # using_optimizers()
    # feeding_data_to_algo()
    # saving_restoring_model()
    # using_tensorboard()
    # name_scopes()
    # modularity()
    sharing_variables()