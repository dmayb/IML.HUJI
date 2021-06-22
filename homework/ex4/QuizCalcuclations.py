import math
import numpy as np

if __name__ == "__main__":
    D = np.array([0.1885, 0.52406, 0.00154, 0.04095, 0.24496])
    ht = np.array([1, -1, -1, 1, -1])
    yi = np.array([1, 1, -1, -1, -1])
    X = np.array([1, 2, 3, 4, 5])

    sames = np.array(yi != ht)
    et = D @ sames
    wt = 0.5 * np.log((1 - et) / et)
    denominator = D @ np.exp((-1) * wt * yi * ht)
    # as is
    nominator = D * np.exp((-1) * wt * yi * ht)
    D4 = nominator / denominator
    print(D4)
