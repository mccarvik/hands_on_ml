import os, pdb
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/4ch/'


def stoch_grad_desc():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    
    theta_path_sgd = []
    m = len(X_b)
    np.random.seed(42)
    
    n_epochs = 50
    t0, t1 = 5, 50  # learning schedule hyperparameters
    
    def learning_schedule(t):
        return t0 / (t + t1)
    
    theta = np.random.randn(2,1)  # random initialization

    for epoch in range(n_epochs):
        for i in range(m):
            if epoch == 0 and i < 20:                    # not shown in the book
                y_predict = X_new_b.dot(theta)           # not shown
                style = "b-" if i > 0 else "r--"         # not shown
                plt.plot(X_new, y_predict, style)        # not shown
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients
            theta_path_sgd.append(theta)                 # not shown
    
    plt.plot(X, y, "b.")                                 # not shown
    plt.xlabel("$x_1$", fontsize=18)                     # not shown
    plt.ylabel("$y$", rotation=0, fontsize=18)           # not shown
    plt.axis([0, 2, 0, 15])                              # not shown
    plt.savefig(PNG_PATH + "sgd_plot.png", dpi=300)      # not shown
    plt.close()                                          # not shown


def grad_desc():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    
    eta = 0.1
    n_iterations = 1000
    m = 100
    theta = np.random.randn(2,1)
    
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
    print(theta)
    
    np.random.seed(42)
    theta = np.random.randn(2,1)  # random initialization
    theta_path_bgd = []

    plt.figure(figsize=(10,4))
    plt.subplot(131); plot_gradient_descent(theta, 0.02, X, y)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132); plot_gradient_descent(theta, 0.1, X, y, theta_path=theta_path_bgd)
    plt.subplot(133); plot_gradient_descent(theta, 0.5, X, y)
    plt.savefig(PNG_PATH + "gradient_descent_plot.png", dpi=300)
    plt.close()


def lin_reg():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(theta_best)
    
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    y_predict = X_new_b.dot(theta_best)
    print(y_predict)
    
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.plot(X_new, y_predict, "r-")
    plt.plot(X, y, "b.")
    plt.axis([0, 2, 0, 15])
    plt.savefig(PNG_PATH + "generated_data_plot.png", tight_layout=False, dpi=300)
    plt.close()
    
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    print(lin_reg.predict(X_new))


def plot_gradient_descent(theta, eta, X, y, theta_path=None):
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)






if __name__ == '__main__':
    # lin_reg()
    # grad_desc()
    stoch_grad_desc()