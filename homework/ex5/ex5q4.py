import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
np.random.seed = 0

domain = (-3.2, 2.2)
m = 1500

def f(x): return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)


def noise(sigma, size): return np.random.normal(0, sigma, size)


def draw_samples(sigma):
    samples = np.random.uniform(domain[0], domain[1], m)
    y = np.apply_along_axis(f, 0, samples)
    y += noise(sigma, m)
    plt.figure()
    plt.plot(samples, y, '*')
    plt.savefig("generatedData_s=" + str(sigma))
    DX = samples[:1000]
    Dy = y[:1000]
    TX = samples[1000:]
    Ty = y[1000:]
    return DX, Dy, TX, Ty


def KFold_cross_validation(k, DX, Dy, sigma):
    y = Dy
    kf = KFold(n_splits=k)
    split_index = list(kf.split(DX))
    degrees = np.arange(1, 15 + 1)
    errors_validation = np.empty(degrees.shape[0])
    errors_train = np.empty(degrees.shape[0])
    for i, degree in enumerate(degrees):
        A = LinearRegression(fit_intercept=degree)
        X = PolynomialFeatures(degree).fit_transform(DX.reshape(-1, 1))
        error_validation = 0
        error_train = 0
        for s, si in split_index:
            A.fit(X[s], y[s])
            y_predicted_validation = A.predict(X[si])
            y_predicted_train = A.predict(X[s])
            error_train += np.sqrt(np.sum((y[s] - y_predicted_train) ** 2))
            error_validation += np.sqrt(np.sum((y[si] - y_predicted_validation)**2))
        errors_validation[i] = error_validation / k
        errors_train[i] = error_train / k
    plt.figure()
    plt.plot(degrees, errors_validation, '-^', label="validation")
    plt.plot(degrees, errors_train, '-^', label="train")
    plt.legend()
    plt.suptitle("loss of KFold validation&training K=" + str(k) + " sigma=" + str(sigma))
    plt.savefig("polynomialKFold" + str(k) + "_s=" + str(sigma))
    return degrees[np.argmin(errors_validation)]


def q4(sigma, ks):
    print("sigma =", sigma)
    DX, Dy, TX, Ty = draw_samples(sigma)
    for k in ks:
        bestDeg = KFold_cross_validation(k, DX, Dy, sigma)
        print("best degree k= " + str(k) + ": ", bestDeg)
        A = LinearRegression(fit_intercept=bestDeg)
        DX_fit = PolynomialFeatures(bestDeg).fit_transform(DX.reshape(-1, 1))
        TX_fit = PolynomialFeatures(bestDeg).fit_transform(TX.reshape(-1, 1))
        A.fit(DX_fit, Dy)
        y_predicted = A.predict(TX_fit)
        error = np.sqrt(np.sum((y_predicted - Ty) ** 2))
        print("K=" + str(k), " test loss is: ", error)


if __name__ == '__main__':
    q4(1, [2, 5])
    q4(5, [2, 5])
