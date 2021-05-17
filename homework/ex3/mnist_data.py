from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from models import Logistic, SVM, DecisionTree, score
from time import time
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def mnist_data_zeros_ones_only():
    mnist = MNIST('dataset\\MNIST')
    x_train, y_train = mnist.load_training()  # 60000 samples
    x_test, y_test = mnist.load_testing()     # 10000 samples
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.int32)
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]
    return x_train, y_train, x_test, y_test


def question12(x_train, y_train):
    x_train = x_train.reshape((x_train.shape[0], 28, 28))
    ones = x_train[y_train == 1]
    zeros = x_train[y_train == 0]
    fig = plt.figure()
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(ones[i], cmap='gray', vmin=0, vmax=255)
        ax = fig.add_subplot(2, 3, i+4)
        ax.imshow(zeros[i], cmap='gray', vmin=0, vmax=255)
    plt.show()
    fig.savefig("q12.png", bbox_inches='tight', pad_inches=0)


def rearrange_data(X):
    return X.reshape((X.shape[0], 784))


def draw_points_until_two_classes(X,y, m):
    labels = np.array([0, 1])
    while True:
        idx = np.random.randint(0, x_train.shape[0], size=m)
        if (np.unique(y[idx]) == labels).all():
            return X[idx], y[idx]


class NearestNeighbors:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        scores = dict()
        predicted_y = self.predict(X)
        if predicted_y is None:
            return scores
        return score(y, predicted_y)


def question14(x_train, y_train, x_test, y_test):
    all_m = [50, 100, 300, 500]
    repeat = 50
    n_neighbors = 4  # after checking some values, this turned out to be the best one
    max_depth = 8    # after checking some values, this turned out to be the best one
    models = [Logistic(), SVM(), DecisionTree(max_depth=max_depth), NearestNeighbors(n_neighbors=n_neighbors)]
    models_names = ["Logistic", "SVM", "Decision Tree, depth="+str(max_depth), "Nearest Neighbors, neighbors="+str(n_neighbors)]

    # models_names = ["Logistic", "SVM"]
    # models = [Logistic(), SVM()]
    # for i in range(3,9):
    #     models.append(DecisionTree(max_depth=i))
    #     models_names.append(["decision " + str(i)])
    #     models.append(NearestNeighbors(n_neighbors=i))
    #     models_names.append(["Neighbors " + str(i)])

    mean_accuracies = np.empty((len(models), len(all_m)))
    accuracies = np.empty((len(models), repeat))
    running_time = np.zeros((len(models), len(all_m)))
    Z, yz = x_test, y_test  # test set
    for i, m in enumerate(all_m):
        for j in range(repeat):
            X, yx = draw_points_until_two_classes(x_train, y_train, m)  # training set
            for k, model in enumerate(models):
                start = time()
                model.fit(X, yx)
                accuracies[k, j] = model.model.score(Z, yz)
                end = time()
                running_time[k, i] += (end - start)
        mean_accuracies[:, i] = accuracies.mean(axis=1)

    colors = ['blue', 'red', 'green', 'orange']
    fig = plt.figure()

    for i, model in enumerate(models_names):
        # plt.plot(all_m, mean_accuracies[i, :], color=colors[i], label=model)
        plt.plot(all_m, mean_accuracies[i, :], label=model)

    plt.legend()
    fig.suptitle("Q14: mean accuracy as function of m")
    fig.savefig("q14.png", bbox_inches='tight', pad_inches=0.1, dpi=fig.dpi)

    running_time = running_time / repeat
    print(pd.DataFrame(running_time, models_names, ["m="+str(m) for m in all_m]))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = mnist_data_zeros_ones_only()
    # question12(x_train, y_train)
    question14(x_train, y_train, x_test, y_test)


