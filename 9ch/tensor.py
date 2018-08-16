import os, pdb, time, sys, time
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/9ch/'


def using_optimizers():
    scaler = StandardScaler()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01
    
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
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
    pdb.set_trace()
    n_epochs = 1000
    learning_rate = 0.01
    
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
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
    pdb.set_trace()
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
    using_auto_diff()