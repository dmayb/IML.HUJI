import numpy as np
from models import Perceptron, SVM, LDA
import matplotlib.pyplot as plt

# samples distribution:
mean = np.array([0, 0])
cov = np.eye(2)

# true hyperplane
w = np.array([0.3, -0.5])
b = 0.1
a = -w[0] / w[1]

k = 1000

# num of samples
all_m = [5, 10, 15, 25, 70]


def draw_points(m):
    X = np.random.multivariate_normal(mean, cov, m)
    y = np.sign((X@w) + b)
    return X, y


def draw_points_until_two_classes(m):
    labels = np.array([-1, 1])
    while True:
        X, y = draw_points(m)
        if (np.unique(y) == labels).all():
            return X, y


def question9():
    perceptron = Perceptron()
    svm = SVM()
    fig = plt.figure()
    plt.suptitle("Q9: True vs. Perceptron vs. SVM hyperplanes")
    for i, m in enumerate(all_m):
        X, y = draw_points_until_two_classes(m)
        svm.fit(X, y)

        ax = fig.add_subplot(2, 3, i+1)
        ax.scatter(X[y == -1, 0], X[y == -1, 1], color="blue", label="y=-1")  # first class, labeled -1
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color="red", label="y=1")     # second class, labeled 1

        xmin, xmax = plt.xlim()
        xx = np.linspace(xmin, xmax)

        true_hyperplane = a * xx - (b / w[1])

        # perceptron
        perceptron.fit(X, y)
        w_perc = perceptron.model[:-1]
        a_perceptron = -w_perc[0] / w_perc[1]
        b_perceptron = perceptron.model[-1] / perceptron.model[1]
        perceptron_hyperplane = a_perceptron * xx - b_perceptron

        w_svm = svm.model.coef_[0]
        a_svm = -w_svm[0] / w_svm[1]
        b_svm = svm.model.intercept_[0] / w_svm[1]
        SVM_hyperplane = a_svm * xx - b_svm

        ax.plot(xx, true_hyperplane, color="black", label="true hyperplane")
        ax.plot(xx, perceptron_hyperplane, color="green", label="perceptron hyperplane")
        ax.plot(xx, SVM_hyperplane, color="orange", label="svm hyperplane")
        ax.title.set_text("m=" + str(m))
        if i == 0:  plt.legend()

    fig.savefig("q9.png", bbox_inches='tight', pad_inches=0.3,  dpi=fig.dpi)


def question10():
    perceptron = Perceptron()
    svm = SVM()
    lda = LDA()
    repeat = 500
    mean_accuracies = np.empty((3, len(all_m)))
    accuracies = np.empty((3, repeat))
    for i, m in enumerate(all_m):
        for j in range(repeat):
            X, yx = draw_points_until_two_classes(m)  # training set
            Z, yz = draw_points_until_two_classes(k)  # test set
            perceptron.fit(X, yx)
            svm.fit(X, yx)
            lda.fit(X, yx)
            accuracies[0, j] = perceptron.score(Z, yz)["accuracy"]
            accuracies[1, j] = svm.score(Z, yz)["accuracy"]
            accuracies[2, j] = lda.score(Z, yz)["accuracy"]
        mean_accuracies[:, i] = accuracies.mean(axis=1)

    models = ["Perceptron", "SVM", "LDA"]
    colors = ['blue', 'red', 'green']
    fig = plt.figure()
    for i, model in enumerate(models):
        plt.plot(all_m, mean_accuracies[i, :], color=colors[i], label=model)
        plt.legend()

    plt.title("Q10: mean accuracy as function of m")
    fig.savefig("q10.png", bbox_inches='tight', pad_inches=0.2, dpi=fig.dpi)


if __name__ == '__main__':
    # question9()
    question10()
