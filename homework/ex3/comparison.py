import numpy as np
from models import Perceptron, SVM
import matplotlib.pyplot as plt

w = np.array([0.3, -0.5])
b = 0.1
a = -w[0] / w[1]


def draw_points(m):
    mean = np.array([0, 0])
    cov = np.eye(2)
    X = np.random.multivariate_normal(mean, cov, m)
    y = np.sign((X@w) + b)
    return X, y


def question9():
    all_m = [5, 10, 15, 25, 70]
    perceptron = Perceptron()
    svm = SVM()
    fig = plt.figure()
    for i, m in enumerate(all_m):
        X, y = draw_points(m)
        perceptron.fit(X, y)
        svm.fit(X, y)

        ax = fig.add_subplot(2, 3, i+1)
        ax.scatter(X[y == -1, 0], X[y == -1, 1], color="blue", label="y=-1")  # first class, labeled -1
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color="red", label="y=1")     # second class, labeled 1

        xmin, xmax = plt.xlim()
        xx = np.linspace(xmin, xmax)

        w_perc = perceptron.model[:-1]
        w_svm = svm.model.coef_[0]
        a_perceptron = -w_perc[0] / w_perc[1]
        a_svm = -w_svm[0] / w_svm[1]

        true_hyperplane = a * xx - b
        perceptron_hyperplane = a_perceptron * xx - (perceptron.model[-1] / perceptron.model[1])
        SVM_hyperplane = a_svm * xx - (svm.model.intercept_[0]) / w_svm[1]

        ax.plot(xx, true_hyperplane, color="black", label="true hyperplane")
        ax.plot(xx, perceptron_hyperplane, color="green", label="perceptron hyperplane")
        ax.plot(xx, SVM_hyperplane, color="orange", label="svm hyperplane")



        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

    # plt.legend(["y=-1", "y=1", "true hyperplane", "perceptron hyperplane", "svm hyperplane"])
    plt.legend()
    fig.savefig("q9.png", bbox_inches='tight',pad_inches=0)


if __name__ == '__main__':
    question9()