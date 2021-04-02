import numpy as np
import matplotlib.pyplot as plt


def createData(epsilons, NTOSSES):
    data = np.empty((len(epsilons), NTOSSES))
    for i, epsilon in enumerate(epsilons):
        data[i, :] = np.random.binomial(1, epsilon, (1, NTOSSES))
    return data


def a():
    epsilon = 0.25
    N_SAMPLES = 5
    N_TOSSES = 1000
    data = np.random.binomial(1, epsilon, (N_SAMPLES, N_TOSSES))
    # meanData = np.empty((N_SAMPLES, N_TOSSES))
    dataCumsum = data.cumsum(axis=1)
    m = np.arange(1, N_TOSSES+1)
    meanData = dataCumsum/m
    # for i in range(N_SAMPLES):
    #     for j in range(N_TOSSES):
    #         meanData[i, j] = np.mean(data[i, 0:j+1])

    fig = plt.figure()
    plt.suptitle("The mean as function of m for X~Ber(0.25)")
    for i in range(N_SAMPLES):
        ax = fig.add_subplot(N_SAMPLES, 1, i+1)
        ax.plot(m, meanData[i, :])
        ax.set_ylim(0, 1)
        ax.tick_params(labelbottom=False)
    ax.tick_params(labelbottom=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    fig.show()
    fig.savefig("16a.png")
    print()


def b():
    N_TOSSES = 1000
    N_SAMPLES = 10000
    p = 0.25  # for c
    pMean = 0.25
    epsilons = [0.5, 0.25, 0.1, 0.01, 0.001]
    data = np.random.binomial(1, p, (N_SAMPLES, N_TOSSES))

    chebyshev = np.empty((len(epsilons), N_TOSSES))
    hoeffding = np.empty((len(epsilons), N_TOSSES))

    varUpperBound = 0.25  # var(x) = p(1-p) <= 0.25 for all 0<=p<=1

    for i, epsilon in enumerate(epsilons):
        epsilon2 = epsilon**2
        for j in range(N_TOSSES):  # m=j+1
            # chebyshev[i, j] = (np.var(data[i, 0:j + 1]))/((j+1)*epsilon)
            chebyshev[i, j] = np.min([varUpperBound / ((j + 1) * epsilon2), 1])

            expArg = (-2*(j+1))*epsilon2
            hoeffding[i, j] = np.min([2*np.exp(expArg), 1])

    # ~~~~~~ C addition ~~~~~~ #
    meanData = np.zeros((N_SAMPLES, N_TOSSES))
    meanData[:, 0] = data[:, 0]
    epsilonPercent = np.zeros((len(epsilons), N_TOSSES))

    for i in range(N_SAMPLES):
        accuracy = np.abs(meanData[i, 0] - pMean)
        epsilonPercent[accuracy >= epsilons, 0] += 1
        for j in range(1,N_TOSSES):
            meanData[i,j] = ((meanData[i,j-1]*j) + data[i, j]) / (j+1)
            accuracy = np.abs(meanData[i,j] - pMean)
            epsilonPercent[accuracy >= epsilons, j] += 1
            pass
    epsilonPercent = epsilonPercent / N_SAMPLES

    m = np.arange(1, N_TOSSES+1)

    # # B
    # for i, epsilon in enumerate(epsilons):
    #     fig = plt.figure()
    #     # fig.show()
    #     plt.suptitle("epsilon: " + str(epsilon))
    #     ax1 = fig.add_subplot(2, 1, 1)
    #     plt.title("chebyshev upper bound as function of m")
    #     ax1.plot(m, chebyshev[i, :])  # plot chevishev
    #
    #     ax2 = fig.add_subplot(2, 1, 2)
    #     plt.title("Hoeffding upper bound as function of m")
    #     ax2.plot(m, hoeffding[i, :], color="red")  # plot chevishev
    #
    #     mng = plt.get_current_fig_manager()
    #     mng.window.showMaximized()
    #
    #     fig.savefig("16b_eps" + str(epsilon) + ".png")


    # C
    for i, epsilon in enumerate(epsilons):
        fig = plt.figure()
        plt.suptitle("epsilon: " + str(epsilon))

        color = "tab:blue"
        ax1 = fig.add_subplot(2, 1, 1)
        plt.title("chebyshev upper bound as function of m")
        ax1.plot(m, chebyshev[i, :],  color=color)  # plot chevishev
        ax1.set_ylabel('upper bound', color=color)
        ax1.tick_params(axis='y', labelcolor=color)


        color = "tab:red"
        ax2 = fig.add_subplot(2, 1, 2)
        plt.title("Hoeffding upper bound as function of m")
        ax2.plot(m, hoeffding[i, :], color=color)  # plot chevishev
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('upper bound', color=color)


        ax3 = ax1.twinx()
        color = "tab:green"
        ax3.set_ylabel('percent', color=color)  # we already handled the x-label with ax1
        ax3.plot(m, epsilonPercent[i,:], color=color)
        ax3.tick_params(axis='y', labelcolor=color)


        ax4 = ax2.twinx()
        color = "tab:orange"
        ax4.set_ylabel('percent', color=color)  # we already handled the x-label with ax1
        ax4.plot(m, epsilonPercent[i,:], color=color)
        ax4.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        fig.savefig("16c_eps" + str(epsilon) + ".png")


# b()
a()