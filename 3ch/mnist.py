import os, pdb
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/3ch/'


def setup():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]
    # some_digit = X[36000]
    # some_digit_image = some_digit.reshape(28, 28)
    # plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    # plt.axis("off")
    # plt.savefig(PNG_PATH + "some_digit_plot.png", dpi=300)
    # print(y[36000])
    
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    # binary_classifier(X_train, X_test, y_train, y_test)
    # roc_curve_demo(X_train, X_test, y_train, y_test)
    # multiclass(X_train, X_test, y_train, y_test)
    multiclass_output(X_train, X_test, y_train, y_test)
 
 
def multiclass_output(X_train, X_test, y_train, y_test):
    some_digit = X_train[20000]
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]

    knn_clf = KNeighborsClassifier()
    # knn_clf.fit(X_train, y_multilabel)
    # print(knn_clf.predict([some_digit]))
    # y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
    # print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))

    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train
    y_test_mod = X_test
    
    some_index = 5500
    plt.subplot(121); plot_digit(X_test_mod[some_index])
    plt.subplot(122); plot_digit(y_test_mod[some_index])
    plt.savefig(PNG_PATH + "noisy_digit_example_plot.png", tight_layout=False, dpi=300)
    plt.close()

    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_test_mod[some_index]])
    plot_digit(clean_digit)
    plt.savefig(PNG_PATH + "cleaned_digit_example_plot.png", tight_layout=False, dpi=300)
    plt.close()
 

def multiclass(X_train, X_test, y_train, y_test):
    some_digit = X_train[20000]
    sgd_clf = SGDClassifier(max_iter=5, random_state=42)
    # sgd_clf.fit(X_train, y_train)
    # print(sgd_clf.predict([some_digit]))
    # print(sgd_clf.decision_function([some_digit]))
    
    # ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
    # ovo_clf.fit(X_train, y_train)
    # print(ovo_clf.predict([some_digit]))
    # print(len(ovo_clf.estimators_))
    
    # forest_clf = RandomForestClassifier(random_state=42)
    # forest_clf.fit(X_train, y_train)
    # print(forest_clf.predict([some_digit]))
    # print(forest_clf.predict_proba([some_digit]))
    # print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
    
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    # print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
    
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
    # y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)

    # plt.matshow(conf_mx, cmap=plt.cm.gray)
    # plt.savefig(PNG_PATH + "confusion_matrix_plot.png", tight_layout=False, dpi=300)
    # plt.close()
    
    # row_sums = conf_mx.sum(axis=1, keepdims=True)
    # norm_conf_mx = conf_mx / row_sums
    # np.fill_diagonal(norm_conf_mx, 0)
    # plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    # plt.savefig(PNG_PATH + "confusion_matrix_errors_plot.png", tight_layout=False, dpi=300)
    # plt.close()
    
    cl_a, cl_b = 3, 5
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
    
    plt.figure(figsize=(8,8))
    plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
    plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
    plt.savefig(PNG_PATH + "error_analysis_digits_plot.png", dpi=300)
    plt.close()
    

def roc_curve_demo(X_train, X_test, y_train, y_test):
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    
    sgd_clf = SGDClassifier(max_iter=5, random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    
    # tpr = recall
    # fpr = FP / (FP + TN)
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr)
    plt.savefig(PNG_PATH + "roc_curve_plot" + ".png", dpi=300)
    plt.close()
    
    print(roc_auc_score(y_train_5, y_scores))
    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig(PNG_PATH + "roc_curve_comparison_plot" + ".png", dpi=300)
    plt.close()
    print(roc_auc_score(y_train_5, y_scores_forest))
    y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
    print(precision_score(y_train_5, y_train_pred_forest))
    print(recall_score(y_train_5, y_train_pred_forest))
    
    
def binary_classifier(X_train, X_test, y_train, y_test):
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(max_iter=5, random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    # print(sgd_clf.predict([X_train[36000]]))
    # print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
    # never_5_clf = Never5Classifier()
    # print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
    
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    print(confusion_matrix(y_train_5, y_train_pred))
    # [[true neg, false pos]
    # [false neg, true pos]]
    # NOTES
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # F1 score = harmonic mean of precision and recall
    print("precision: " + str(precision_score(y_train_5, y_train_pred)))
    print("AKA when it predicts a 5 it is only right this percent")
    print("recall: " + str(recall_score(y_train_5, y_train_pred)))
    print("AKA only accurately detects this percent of the 5s")
    print("f1 score: " + str(f1_score(y_train_5, y_train_pred)))
    
    # Setting custom threshold
    # y_scores = sgd_clf.decision_function([some_digit])
    # threshold = 0
    # y_some_digit_pred = (y_scores > threshold)
    # print(y_some_digit_pred)
    # threshold = 200000
    # y_some_digit_pred = (y_scores > threshold)
    # print(y_some_digit_pred)
    
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    # hack to work around issue #9589 introduced in Scikit-Learn 0.19.0
    if y_scores.ndim == 2:
        y_scores = y_scores[:, 1]

    # precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    # plt.figure(figsize=(8, 4))
    # plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    # plt.xlim([-700000, 700000])
    # plt.savefig(PNG_PATH + "precision_recall_vs_threshold_plot" + ".png", dpi=300)
    # plt.close()
    
    # plt.figure(figsize=(8, 6))
    # plot_precision_vs_recall(precisions, recalls)
    # plt.savefig(PNG_PATH + "precision_vs_recall_plot" + ".png", dpi=300)
    # plt.close()
    
    y_train_pred_90 = (y_scores > 70000)
    print(precision_score(y_train_5, y_train_pred_90))
    print(recall_score(y_train_5, y_train_pred_90))


def custom_folds(X_train, y_train_5):
    # pseudo custom way to have more control over the kfolds process
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train_5[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train_5[test_index])
    
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))
    

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")



if __name__ == '__main__':
    setup()