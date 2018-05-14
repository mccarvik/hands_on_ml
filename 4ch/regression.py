import os, pdb
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/4ch/'


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
    (array([4.21509616]), array([[2.77011339]]))
    print(lin_reg.predict(X_new))



if __name__ == '__main__':
    lin_reg()