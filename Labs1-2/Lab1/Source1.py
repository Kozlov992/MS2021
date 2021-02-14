from scipy.stats import norm, laplace, poisson, cauchy, uniform
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

units = [10, 50, 1000]


def norm_numbers():
    bins_num = [8, 12, 20]
    default_left_boundary = -3
    default_right_boundary = 3
    fig, axs = plt.subplots(len(units))
    for i in range(len(units)):
        samples = norm.rvs(scale=1, loc=0, size=units[i])
        left_boundary = min(default_left_boundary, min(samples))
        right_boundary = max(default_right_boundary, max(samples))
        axs[i].grid()
        sns.histplot(samples, stat="density", bins=bins_num[i], color='skyblue', ax=axs[i])
        x = np.linspace(left_boundary, right_boundary, 1000)
        y = norm.pdf(x)
        axs[i].plot(x, y, 'k', lw=2)
        axs[i].set_xlabel("normalNumbers (" + str(units[i]) + " samples)")
    fig.subplots_adjust(hspace=0.75)
    fig.savefig("normalNumbers.pdf")
    fig.show()


def cauchy_numbers():
    bins_num = [5, 12, 20]
    default_left_boundary = -5
    default_right_boundary = 5
    fig, axs = plt.subplots(len(units))
    for i in range(len(units)):
        samples = cauchy.rvs(scale=1, loc=0, size=units[i])
        left_boundary = min(default_left_boundary, min(samples))
        right_boundary = max(default_right_boundary, max(samples))
        axs[i].grid()
        sns.histplot(samples, stat="density", bins=bins_num[i], color='sandybrown', ax=axs[i])
        x = np.linspace(left_boundary, right_boundary, 1000)
        y = cauchy.pdf(x)
        axs[i].plot(x, y, 'k', lw=2)
        axs[i].set_xlabel("cauchyNumbers (" + str(units[i]) + " samples)")
    fig.subplots_adjust(hspace=0.75)
    fig.savefig("cauchyNumbers.pdf")
    fig.show()


def laplace_numbers():
    bins_num = [8, 12, 20]
    default_left_boundary = -5
    default_right_boundary = 5
    fig, axs = plt.subplots(len(units))
    for i in range(len(units)):
        samples = laplace.rvs(scale=2 ** (-0.5), loc=0, size=units[i])
        left_boundary = min(default_left_boundary, min(samples))
        right_boundary = max(default_right_boundary, max(samples))
        axs[i].grid()
        sns.histplot(samples, stat="density", bins=bins_num[i], color='salmon', ax=axs[i])
        x = np.linspace(left_boundary, right_boundary, 1000)
        y = laplace(scale=2 ** (-0.5), loc=0).pdf(x)
        axs[i].plot(x, y, 'k', lw=2)
        axs[i].set_xlabel("laplaceNumbers (" + str(units[i]) + " samples)")
    fig.subplots_adjust(hspace=0.75)
    fig.savefig("laplaceNumbers.pdf")
    fig.show()

    
def poisson_numbers():
    bins_num = [8, 12, 20]
    default_right_boundary = 10
    fig, axs = plt.subplots(len(units))
    for i in range(len(units)):
        samples = poisson.rvs(10, size=units[i])
        left_boundary = 0.
        right_boundary = max(default_right_boundary, max(samples))
        axs[i].grid()
        sns.histplot(samples, stat="density", bins=bins_num[i], color='seagreen', ax=axs[i])
        x = np.linspace(left_boundary, right_boundary, right_boundary + 1)
        y = poisson(10).pmf(x)
        axs[i].plot(x, y, 'k', lw=2)
        axs[i].set_xlabel("poissonNumbers (" + str(units[i]) + " samples)")
    fig.subplots_adjust(hspace=0.75)
    fig.savefig("poissonNumbers.pdf")
    fig.show()

    
def uniform_numbers():
    bins_num = [8, 12, 30]
    fig, axs = plt.subplots(len(units))
    for i in range(len(units)):
        samples = uniform.rvs(loc=-3 ** 0.5, scale=2 * 3 ** 0.5, size=units[i])
        axs[i].grid()
        sns.histplot(samples, stat="density", bins=bins_num[i], color='slateblue', ax=axs[i])
        x = np.linspace(-2, 2, 1000)
        y = uniform(loc=-3 ** 0.5, scale=2 * 3 ** 0.5).pdf(x)
        axs[i].plot(x, y, 'k', lw=2)
        axs[i].set_xlabel("uniformNumbers (" + str(units[i]) + " samples)")
    fig.subplots_adjust(hspace=0.75)
    fig.savefig("uniformNumbers.pdf")
    fig.show()
    
    
