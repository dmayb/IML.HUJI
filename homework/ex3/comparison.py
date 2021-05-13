import numpy as np
from models import Perceptron, SVM
import matplotlib.pyplot as plt


def draw_points(m):
    mean = np.array([0, 0])
    cov = np.eye(2)
    X = np.random.multivariate_normal(mean, cov, m)
    w = np.array([0.3, -0.5])
    y = np.sign((X@w) + 0.1)
    return X, y


def question9():
    all_m = [5, 10, 15, 25, 70]
    perceptron = Perceptron()
    svm = SVM()
    fig = plt.figure()
    for i, m in enumerate(all_m):
        X, y = draw_points(m)
        perceptron.fit(X, y)
        y_prediction_perceptron = perceptron.predict(X)
        svm.fit(X, y)
        y_prediction_svm = perceptron.predict(X)
        results = [y, y_prediction_svm, y_prediction_svm]
        models_names = ["True Values", "Perceptron", "SVM"]

        ax = fig.add_subplot(2, 3, i)
        ax.scatter(X[y == -1, 0], X[y == -1, 1], color="blue")  # first class, labeled -1
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color="red")     # second class, labeled 1
        


