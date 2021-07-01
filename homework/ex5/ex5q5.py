from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import numpy as np
import matplotlib.pyplot as plt
k = 5


def KFold_cross_validation(split_index, model, lambdaStart, lambdaEnd, num_lambdas,  X, y):
    lambdas = np.linspace(lambdaStart, lambdaEnd, num=num_lambdas)
    errors_validation = np.empty(lambdas.shape[0])
    errors_train = np.empty(lambdas.shape[0])
    for i, lamda in enumerate(lambdas):
        A = model(alpha=lamda)
        error_validation = 0
        error_train = 0
        for s, si in split_index:
            A.fit(X[s], y[s])
            y_predicted_validation = A.predict(X[si])
            y_predicted_train = A.predict(X[s])
            error_train += np.mean((y[s] - y_predicted_train) ** 2)
            error_validation += np.mean((y[si] - y_predicted_validation) ** 2)
        errors_validation[i] = error_validation / k
        errors_train[i] = error_train / k
    modelName = str(A)[:5]
    plt.plot(lambdas, errors_validation, label="validation " + modelName)
    plt.plot(lambdas, errors_train, label="train " + modelName)
    return lambdas[np.argmin(errors_validation)]


if __name__ == '__main__':
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train = X[:50, :], y[:50]
    X_test, y_test = X[50:, :], y[50:]
    kf = KFold(n_splits=k)
    split_index = list(kf.split(X_train))
    l_ridge=KFold_cross_validation(split_index, Ridge, 0.001, 2, 50, X_train, y_train)
    print("best lambda for ridge: " + str(l_ridge))
    l_lasso=KFold_cross_validation(split_index, Lasso, 0.001, 2, 50, X_train, y_train)
    print("best lambda for lasso: " + str(l_lasso))
    plt.legend()
    plt.suptitle("loss of KFold validation&training ")
    plt.savefig("linearKFold")

    models = [Ridge(alpha=l_ridge), Lasso(alpha=l_lasso), LinearRegression()]
    modelsNames = ['Ridge', 'Lasso', 'LinearRegression']
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)
        error = np.mean((y_test-y_predicted)**2)
        print(modelsNames[i], "test error: ", error)
