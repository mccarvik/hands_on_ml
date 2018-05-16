import os, pdb
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn import datasets
from matplotlib.colors import ListedColormap

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/4ch/'


def softmax():
    # logistic regression generalizes to support multiple regression
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]
    
    softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
    softmax_reg.fit(X, y)
    x0, x1 = np.meshgrid(
            np.linspace(0, 8, 500).reshape(-1, 1),
            np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    
    y_proba = softmax_reg.predict_proba(X_new)
    y_predict = softmax_reg.predict(X_new)
    
    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)
    
    plt.figure(figsize=(10, 4))
    plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
    plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolor")
    plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 7, 0, 3.5])
    plt.savefig(PNG_PATH + "softmax_regression_contour_plot.png", dpi=300)
    plt.close()
    
    print(softmax_reg.predict([[5, 2]]))
    print(softmax_reg.predict_proba([[5, 2]]))
    


def logistic_regr():
    # uses a sigmoid function to output a 0 or 1 for a given sample of values using the weight vector and input values as paramters
    # sigmoid(t) = 1 / (1 + e**(-t))
    
    # Sigmoid
    t = np.linspace(-10, 10, 100)
    sig = 1 / (1 + np.exp(-t))
    plt.figure(figsize=(9, 3))
    plt.plot([-10, 10], [0, 0], "k-")
    plt.plot([-10, 10], [0.5, 0.5], "k:")
    plt.plot([-10, 10], [1, 1], "k:")
    plt.plot([0, 0], [-1.1, 1.1], "k-")
    plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    plt.xlabel("t")
    plt.legend(loc="upper left", fontsize=20)
    plt.axis([-10, 10, -0.1, 1.1])
    plt.savefig(PNG_PATH + "logistic_function_plot.png", dpi=300)
    plt.close()
    
    iris = datasets.load_iris()
    print(list(iris.keys()))
    # print(iris.DESCR)
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X, y)
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)

    plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

    plt.figure(figsize=(8, 3))
    plt.plot(X[y==0], y[y==0], "bs")
    plt.plot(X[y==1], y[y==1], "g^")
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
    plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
    plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
    plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
    plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
    plt.xlabel("Petal width (cm)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 3, -0.02, 1.02])
    plt.savefig(PNG_PATH + "logistic_regression_plot.png", dpi=300)
    plt.close()

    print(decision_boundary)
    print(log_reg.predict([[1.7], [1.5]]))
    
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.int)

    log_reg = LogisticRegression(C=10**10, random_state=42)
    log_reg.fit(X, y)

    x0, x1 = np.meshgrid(
            np.linspace(2.9, 7, 500).reshape(-1, 1),
            np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_proba = log_reg.predict_proba(X_new)
    
    plt.figure(figsize=(10, 4))
    plt.plot(X[y==0, 0], X[y==0, 1], "bs")
    plt.plot(X[y==1, 0], X[y==1, 1], "g^")
    
    zz = y_proba[:, 1].reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
    left_right = np.array([2.9, 7])
    boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
    
    plt.clabel(contour, inline=1, fontsize=12)
    plt.plot(left_right, boundary, "k--", linewidth=3)
    plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
    plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis([2.9, 7, 0.8, 2.7])
    plt.savefig(PNG_PATH + "logistic_regression_contour_plot.png", dpi=300)
    plt.close()


def early_stopping():
    # stop the algorithm learning when validation error stops decreasing, even if testing error keeps decreasing (overfitting)
    np.random.seed(42)
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

    X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

    poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler()),
    ])

    X_train_poly_scaled = poly_scaler.fit_transform(X_train)
    X_val_poly_scaled = poly_scaler.transform(X_val)

    sgd_reg = SGDRegressor(max_iter=1, penalty=None, eta0=0.0005, warm_start=True, learning_rate="constant", random_state=42)

    n_epochs = 500
    train_errors, val_errors = [], []
    for epoch in range(n_epochs):
        sgd_reg.fit(X_train_poly_scaled, y_train)
        y_train_predict = sgd_reg.predict(X_train_poly_scaled)
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        train_errors.append(mean_squared_error(y_train, y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    best_epoch = np.argmin(val_errors)
    best_val_rmse = np.sqrt(val_errors[best_epoch])
    
    plt.annotate('Best model',
                 xy=(best_epoch, best_val_rmse),
                 xytext=(best_epoch, best_val_rmse + 1),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=16,
                )
    
    best_val_rmse -= 0.03  # just to make the graph look better
    plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.savefig(PNG_PATH + "early_stopping_plot.png", dpi=300)
    plt.close()


def regularized():
    # It adds the sum of the weights**2 as part of the cost function
    # So function is encouraged to keep the weights smaller preventing overfitting
    # a = coefficient of the summation, so higher A = more costs for larger weights (more regularized) and vice versa
    np.random.seed(42)
    m = 20
    X = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
    X_new = np.linspace(0, 3, 100).reshape(100, 1)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plot_model(X, y, X_new, Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plot_model(X, y, X_new, Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
    plt.savefig(PNG_PATH + "ridge_regression_plot.png", dpi=300)
    plt.close()
    
    ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))

    sgd_reg = SGDRegressor(max_iter=5, penalty="l2", random_state=42)
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.predict([[1.5]]))

    ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))

    # Lasso similar regularization to Ridge except using absolute value of weights instead of squares
    # Tends to completely eliminate the weights of the least important features
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plot_model(X, y, X_new, Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plot_model(X, y, X_new, Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)
    plt.savefig(PNG_PATH + "lasso_regression_plot.png", dpi=300)
    plt.close()

    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    print(lasso_reg.predict([[1.5]]))
    
    # Elastic Net - middle ground between ridge and lasso, uses half of abs value and half of squares of weights
    # has another parameter, r, that is the mix ratio to decide how much to use of each
    # r=0 --> elastic net = ridge, r=1 --> elastic net = lasso
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic_net.fit(X, y)
    print(elastic_net.predict([[1.5]]))
    

def polynomial():
    np.random.seed(42)
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    print(X[0])
    print(X_poly[0])

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    X_new=np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)
    # plt.plot(X, y, "b.")
    # plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
    # plt.xlabel("$x_1$", fontsize=18)
    # plt.ylabel("$y$", rotation=0, fontsize=18)
    # plt.legend(loc="upper left", fontsize=14)
    # plt.axis([-3, 3, 0, 10])
    # plt.savefig(PNG_PATH + "quadratic_predictions_plot.png", dpi=300)
    # plt.close()
    

    # for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
    #     polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    #     std_scaler = StandardScaler()
    #     lin_reg = LinearRegression()
    #     polynomial_regression = Pipeline([
    #             ("poly_features", polybig_features),
    #             ("std_scaler", std_scaler),
    #             ("lin_reg", lin_reg),
    #         ])
    #     polynomial_regression.fit(X, y)
    #     y_newbig = polynomial_regression.predict(X_new)
    #     plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)
    
    # plt.plot(X, y, "b.", linewidth=3)
    # plt.legend(loc="upper left")
    # plt.xlabel("$x_1$", fontsize=18)
    # plt.ylabel("$y$", rotation=0, fontsize=18)
    # plt.axis([-3, 3, 0, 10])
    # plt.savefig(PNG_PATH + "high_degree_polynomials_plot.png", dpi=300)
    # plt.close()
    
    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)
    plt.axis([0, 80, 0, 3])                         # not shown in the book
    plt.savefig(PNG_PATH + "underfitting_learning_curves_plot.png", dpi=300)   # not shown
    plt.close()                                      # not shown

    polynomial_regression = Pipeline([
            ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
            ("lin_reg", LinearRegression()),
    ])

    plot_learning_curves(polynomial_regression, X, y)
    plt.axis([0, 80, 0, 3])           # not shown
    plt.savefig(PNG_PATH + "learning_curves_plot.png", dpi=300)  # not shown
    plt.close()                        # not shown


def mini_batch_grad_desc():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

    theta_path_mgd = []

    n_iterations = 50
    minibatch_size = 20
    m = 100

    np.random.seed(42)
    theta = np.random.randn(2,1)  # random initialization

    t0, t1 = 200, 1000
    def learning_schedule(t):
        return t0 / (t + t1)

    # mini batch gradient descent
    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            # learns on a batch of "minibatch_size", somwhere between the whole batch and an individual observation
            t += 1
            xi = X_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]
            gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(t)
            theta = theta - eta * gradients
            theta_path_mgd.append(theta)
    print(theta)
    
    
    # stochastic gradient descent
    n_epochs = 50
    t0, t1 = 5, 50  # learning schedule hyperparameters
    def learning_schedule(t):
        return t0 / (t + t1)
    
    theta = np.random.randn(2,1)  # random initialization
    theta_path_sgd = []
    for epoch in range(n_epochs):
        # updates on each one, not in a batch
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients
            theta_path_sgd.append(theta)  


    # batch gradient descent
    eta = 0.1
    n_iterations = 1000
    m = 100
    theta = np.random.randn(2,1)
    theta_path_bgd = [theta]
    
    for iteration in range(n_iterations):
        # updates once for whole batch
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        theta_path_bgd.append(theta)
    print(theta)
    
    
    theta_path_bgd = np.array(theta_path_bgd)
    theta_path_sgd = np.array(theta_path_sgd)
    theta_path_mgd = np.array(theta_path_mgd)

    plt.figure(figsize=(7,4))
    plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
    plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
    plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
    plt.legend(loc="upper left", fontsize=16)
    plt.xlabel(r"$\theta_0$", fontsize=20)
    plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
    plt.axis([2.5, 4.5, 2.3, 3.9])
    plt.savefig(PNG_PATH + "gradient_descent_paths_plot.png", dpi=300)
    plt.close()


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
        # updates on each one, not in a batch
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

    sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.intercept_, sgd_reg.coef_)


def grad_desc():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    
    eta = 0.1
    n_iterations = 1000
    m = 100
    theta = np.random.randn(2,1)
    
    for iteration in range(n_iterations):
        # updates once for whole batch
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
    

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown


def plot_model(X, y, X_new, model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])




if __name__ == '__main__':
    # lin_reg()
    # grad_desc()
    # stoch_grad_desc()
    # mini_batch_grad_desc()
    # polynomial()
    # regularized()
    # early_stopping()
    # logistic_regr()
    softmax()