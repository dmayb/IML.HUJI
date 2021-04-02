import numpy as np
import matplotlib.pyplot as plt


def createData(epsilons, NTOSSES):
    data = np.empty((len(epsilons), NTOSSES))
    for i, epsilon in enumerate(epsilons):
        data[i, :] = np.random.binomial(1, epsilon, (1, NTOSSES))
    return data


def b():
    N_TOSSES = 1000
    epsilons = [0.5, 0.25, 0.1, 0.01, 0.001]
    data = createData(epsilons, N_TOSSES)
    chebyshev = np.empty((len(epsilons), N_TOSSES))
    hoeffding = np.empty((len(epsilons), N_TOSSES))

    for i, epsilon in enumerate(epsilons):
        var = epsilon*(1-epsilon)
        epsilon2 = epsilon**2
        for j in range(N_TOSSES):  # m=j+1
            # chebyshev[i, j] = (np.var(data[i, 0:j + 1]))/((j+1)*epsilon)
            chebyshev[i, j] = var / ((j + 1) * epsilon)
            if chebyshev[i, j] > 1:
                chebyshev[i, j] = chebyshev[i, j] - 1

            expArg = ((-2*((j+1)**2))*epsilon2)
            hoeffding[i, j] = 2*np.exp(expArg)

    m = np.arange(1, N_TOSSES+1)
    for i, epsilon in enumerate(epsilons):
        fig = plt.figure()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.title("epsilon: " + str(epsilon))
        ax = fig.add_subplot(2, 1, 1)
        plt.title("chebyshev upper bound as function of m")
        ax.plot(m, chebyshev[i, :])  # plot chevishev
        # ax.set_ylim(0, 0.4)

        ax = fig.add_subplot(2, 1, 2)
        plt.title("Hoeffding upper bound as function of m")
        ax.plot(m, hoeffding[i, :])  # plot chevishev
        # ax.set_ylim(0, 0.4)


        fig.show()

        fig.savefig("16b_eps" + str(epsilon) + ".png")

b()