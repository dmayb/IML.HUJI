import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def score(y, predicted_y):
    FP = np.sum(predicted_y > y)
    TP = np.sum((predicted_y == y) & predicted_y == 1)
    FN = np.sum(predicted_y < y)
    TN = np.sum((predicted_y == y) & predicted_y == -1)
    P = TP + FN
    N = TN + FP
    scores = dict()
    scores["num_of_samples"] = P + N
    scores["error"] = (FP + FN) / (P + N)
    scores["accuracy"] = (TP + TN) / (P + N)
    scores["FPR"] = FP / N
    scores["TPR"] = TP / P
    scores["precision"] = TP / (FP + TP)
    scores["specificity"] = TN / (TN + FP)
    return scores


class Perceptron:
    def __init__(self):
        self.model = []

    def fit(self, X, y):
        """
        Perceptron algorithm for Half-Space classification. Assumes data is linearly separable
        :param X: m x d matrix - m samples, d features
        :param y: m length vector of tags (1 or -1)
        :return: stores w (the halfspace) in self.model
        """
        w = np.zeros(y.shape[0])  # init w
        while True:
            condition = y*(X@w)  # check if there is a mis-labled sample
            wrongLabelIdx = np.where(condition <= 0)[0]
            if wrongLabelIdx.size > 0 :
                w = w + y[wrongLabelIdx[0]] * X[wrongLabelIdx[0], :]
            else:
                self.model = w
                return

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels y
        :param X: unlabeled test set
        :return: predicted labels
        """
        if not self.model:
            return np.sign(X@self.model)

    def score(self, X, y):
        """
        return dict with:
        :param X:
        :param y:
        :return:
        """
        scores = dict()
        predicted_y = self.predict(X)
        if not predicted_y:
            return scores
        return score(y, predicted_y)


class LDA:
    def __init__(self):
        self.model = dict()

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return: saves to self.model a dictionary with keys:
        p_y1, p_ym1, mu1, mu_m1 sigma,  calculated by the estimators in q3
        """

        m = y.shape[0]
        y1 = (y == 1)                                 # where y == 1
        ym1 = (y == -1)                               # where y == -1
        y1_count = np.sum(y1).sum()                   # number of samples labeled 1
        ym1_count = m - y1_count                      # number of samples labeled -1
        self.model["p_y1"] = y1_count / m             # estimated probability for y=1
        self.model["p_ym1"] = 1 - self.model["p_y1"]  # estimated probability for y=-1

        mu1 = X[y1].sum(axis=0) / y1_count      # estimated mean vector for a sample x, given its labeled 1
        mu_m1 = X[ym1].sum(axis=0) / ym1_count  # estimated mean vector for a sample x, given its labeled -1
        self.model["mu1"] = mu1
        self.model["mu_m1"] = mu_m1

        tmp = (X[y1] - mu1)[:, :, np.newaxis]
        sigma_1 = np.matmul(tmp, tmp).sum(axis=0)
        tmp = (X[ym1] - mu_m1)[:, :, np.newaxis]
        sigma_m1 = np.matmul(tmp, tmp).sum(axis=0)
        self.model["sigma"] = (sigma_1 + sigma_m1) / m

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels y
        :param X: unlabeled test set
        :return: predicted labels
        """
        sigma_inverse = np.linalg.inv(self.model["sigma"])
        sigma_inv_mul_mu1 = sigma_inverse@self.model["mu1"]
        delta1 = X@sigma_inv_mul_mu1 - 0.5*self.model["mu"]@sigma_inv_mul_mu1 + np.log(self.model["p_y1"])
        sigma_inv_mul_mu_m1 = sigma_inverse@self.model["mu1"]
        delta_m1 = X@sigma_inv_mul_mu_m1 - 0.5*self.model["mu_m1"]@sigma_inv_mul_mu_m1 + np.log(self.model["p_ym1"])
        return np.where(delta1 >= delta_m1, np.ones(X.shape[0]), np.full(X.shape[0], -1))

    def score(self, X, y):
        scores = dict()
        predicted_y = self.predict(X)
        if not predicted_y:
            return scores
        return score(y, predicted_y)


class SVM:
    def __init__(self):
        self.model = SVC(C=1e10, kernel="linear")

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: fits self.model (which is svm.SVC model)
        """
        self.model.fit(X,y)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels y
        :param X: unlabeled test set
        :return: predicted labels
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        return dict with:
        :param X:
        :param y:
        :return:
        """
        scores = dict()
        predicted_y = self.predict(X)
        if not predicted_y:
            return scores
        return score(y, predicted_y)


class Logistic:
    def __init__(self):
        self.model = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: fits self.model (which is LogisticRegression model)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels y
        :param X: unlabeled test set
        :return: predicted labels
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        return dict with:
        :param X:
        :param y:
        :return:
        """
        scores = dict()
        predicted_y = self.predict(X)
        if not predicted_y:
            return scores
        return score(y, predicted_y)


class DecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=5)

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: fits self.model (which is LogisticRegression model)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels y
        :param X: unlabeled test set
        :return: predicted labels
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        return dict with:
        :param X:
        :param y:
        :return:
        """
        scores = dict()
        predicted_y = self.predict(X)
        if not predicted_y:
            return scores
        return score(y, predicted_y)