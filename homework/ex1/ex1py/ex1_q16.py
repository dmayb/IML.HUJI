import numpy as np
import matplotlib.pyplot as plt


def a(plot=False):
    epsilon = 0.25
    N_SAMPLES = 5
    N_TOSSES = 1000
    data = np.random.binomial(1, epsilon, (N_SAMPLES, N_TOSSES))
    dataCumsum = data.cumsum(axis=1)
    m = np.arange(1, N_TOSSES+1)
    meanData = dataCumsum/m

    if not plot:
        return

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


def b(plot=False):
    N_TOSSES = 1000
    N_SAMPLES = 100000
    p = 0.25  # for c
    epsilons = [0.5, 0.25, 0.1, 0.01, 0.001]
    # data = np.random.binomial(1, p, (N_SAMPLES, N_TOSSES))

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

    if not plot:
        return chebyshev, hoeffding

    m = np.arange(1, N_TOSSES + 1)

    for i, epsilon in enumerate(epsilons):
        fig = plt.figure()
        # fig.show()
        plt.suptitle("epsilon: " + str(epsilon))
        ax1 = fig.add_subplot(2, 1, 1)
        plt.title("chebyshev upper bound as function of m")
        ax1.plot(m, chebyshev[i, :])  # plot chevishev

        ax2 = fig.add_subplot(2, 1, 2)
        plt.title("Hoeffding upper bound as function of m")
        ax2.plot(m, hoeffding[i, :], color="red")  # plot chevishev

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        fig.savefig("16b_eps" + str(epsilon) + ".png")
    return chebyshev, hoeffding


def c(chebyshev, hoeffding, plot=False):
    N_TOSSES = 1000
    N_SAMPLES = 100000
    p = 0.25
    pMean = 0.25
    epsilons = [0.5, 0.25, 0.1, 0.01, 0.001]
    data = np.random.binomial(1, p, (N_SAMPLES, N_TOSSES))

    meanData = np.zeros((N_SAMPLES, N_TOSSES))
    meanData[:, 0] = data[:, 0]
    epsilonPercent = np.empty((len(epsilons), N_TOSSES))
    dataCumsum = data.cumsum(axis=1)
    m = np.arange(1, N_TOSSES + 1)
    meanData = dataCumsum / m
    accuracy = np.abs(meanData - pMean)

    for j in range(0, N_TOSSES):
        for i, epsilon in enumerate(epsilons):
            epsilonPercent[i, j] = (accuracy[:, j] >= epsilon).sum() / N_SAMPLES

    if not plot:
        return

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
        fig.savefig("16c_eps" + str(epsilon) + ".png", bbox_inches='tight',pad_inches=0)


if __name__ == '__main__':
        # a()
        chebyshev, hoeffding = b()
        c(chebyshev, hoeffding, plot=True)
