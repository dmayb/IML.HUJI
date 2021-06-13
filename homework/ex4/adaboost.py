"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np
from ex4_tools import *
import matplotlib.pyplot as plt
import cProfile


def profile_func(func):
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.print_stats()
        return retval

    return wrapper

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = X.shape[0]
        D = np.full(m, 1/m)

        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            prediction = self.h[t].predict(X)
            epsilon = np.sum(D[prediction != y])
            self.w[t] = 0.5 * np.log((1/epsilon) - 1)
            d_exp = D * np.exp(-(self.w[t] * y * prediction))
            D = d_exp / np.sum(d_exp)
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        predictions = np.array([self.h[t].predict(X) for t in range(max_t)]).T
        h_boost = np.sign(np.sum(self.w[:max_t] * predictions, axis=1))
        return h_boost

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        prediction = self.predict(X, max_t)
        error = np.sum(prediction != y) / y.shape[0]
        return error


# @profile_func
def question13_14_15_16(noise):
    T = 500
    X_train, y_train = generate_data(5000, noise)
    adaBoost = AdaBoost(DecisionStump, T)
    D = adaBoost.train(X_train, y_train)
    D = (D / np.max(D)) * 10  # normalize D so we can see it in the plot
    X_test, y_test = generate_data(200, noise)
    train_error = np.empty(T)
    test_error = np.empty(T)
    for t in range(1, T+1):
        train_error[t-1] = adaBoost.error(X_train, y_train, t)
        test_error[t-1] = adaBoost.error(X_test, y_test, t)

    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    x = np.arange(1, T+1)
    plt.plot(x, train_error, label="train error")
    plt.plot(x, test_error, label="test error")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("error ratio")
    # fig.suptitle("Q13: train & test error in AdaBoost as function of T")
    fig.savefig("Q13_noise" + str(noise) + ".png", bbox_inches='tight', pad_inches=0.1, dpi=fig.dpi)

    Ts = np.array([5, 10, 50, 100, 200, 500])
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    for i, t in enumerate(Ts):
        ax = fig.add_subplot(2, 3, i+1)
        decision_boundaries(adaBoost, X_test, y_test, num_classifiers=t)
        ax.title.set_text("T=" + str(t))
    # fig.suptitle("Q14: decision_boundaries of different T's")
    fig.savefig("Q14_noise" + str(noise) + ".png", bbox_inches='tight', pad_inches=0.1, dpi=fig.dpi)

    T_min_error = test_error.argmin() + 1
    print("~~~Q15~~~\n T that minimize the error: " + str(T_min_error))
    print("the error: " + str(test_error[T_min_error-1]))

    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    decision_boundaries(adaBoost, X_train, y_train, num_classifiers=T_min_error)
    # fig.suptitle("Q15: T that min the error, predicting the training set")
    fig.savefig("Q15_noise" + str(noise) + ".png", bbox_inches='tight', pad_inches=0.1, dpi=fig.dpi)

    fig = plt.figure()
    decision_boundaries(adaBoost, X_train, y_train, num_classifiers=T, weights=D)
    # fig.suptitle("Q16: training set decision boundary with D")
    fig.savefig("Q16_noise" + str(noise) + ".png", bbox_inches='tight', pad_inches=0.1, dpi=fig.dpi)


if __name__ == '__main__':
    question13_14_15_16(0)
    question13_14_15_16(0.01)
    question13_14_15_16(0.4)

