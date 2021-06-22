from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, Ridge
import numpy as np
import matplotlib.pyplot as plt
k = 5


def KFold_cross_validation(split_index, model, lambdaStart, lambdaEnd, gap,  X, y):
    lambdas = np.arange(lambdaStart, lambdaEnd, gap)
    errors = np.empty(lambdas.shape[0])
    for i, lamda in enumerate(lambdas):
        A = model(alpha=lamda)
        error = 0
        for s, si in split_index:
            A.fit(X[s], y[s])
            y_predict = A.predict(X[si])
            error += np.sqrt(np.sum((y[si] - y_predict)**2))
        errors[i] = error / k
    plt.plot(lambdas, errors)
    plt.show()
    return lambdas[np.argmin(errors)]

if __name__ == '__main__':
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train = X[:50, :], y[:50]
    X_test, y_test = X[50:, :], y[50:]
    kf = KFold(n_splits=k)
    split_index = list(kf.split(X_train))
    l=KFold_cross_validation(split_index, Ridge, 0.005, 0.05, 0.005, X_train, y_train)
    print("best lambda for ridge: " + str(l))
    l=KFold_cross_validation(split_index, Lasso, 0.005, 0.7, 0.005, X_train, y_train)
    print("best lambda for lasso: " + str(l))