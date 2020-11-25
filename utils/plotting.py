import numpy as np
import matplotlib.pyplot as plt


def moving_average(arr, n=None):
    """
    Calculates the moving average over an array

    Parameters
    ----------
    arr: np.ndarray
        The array over which to calculate the moving average
    n: int, default = len(arr) // 7
        The size of the moving average

    Returns
    -------
    np.ndarray
        Array of the same length as `arr`, which holds the y values of the moving average
    """
    # n is the length of moving average.
    if n is None:
        n = len(arr) // 7

    # Calculate moving average
    cumsum = np.cumsum(arr, dtype=float)
    cumsum[n:] = cumsum[n:] - cumsum[:-n]
    moving_avg = np.zeros(len(cumsum))
    moving_avg[n-1:] = cumsum[n-1:] / n

    # Calculate a scaling moving average for the first n elements.
    for i in range(n):
        moving_avg[i] = cumsum[i] / (i + 1)

    return moving_avg


def plot1(scores, path=None):
    """Plots a single array and save it to path if specified."""
    plt.figure(2)
    plt.clf()
    plt.title('Results')
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.plot(moving_average(scores))

    if path is not None:
        plt.savefig(fname=path)
    else:
        plt.show()


def plot_all(mean, std, n=0, path=None):
    """
    Makes a line-plot which includes standard deviation

    Parameters
    ----------
    mean: np.ndarray
        Array representing the mean results over all experiments
    std: np.ndarray
        Array representing the standard deviation of each result
    n: int
        Number of experiments ran
    path: str, optional
        Where to save the resulting plot. Shows the plot instead if
        no path has been given.
    """
    plt.figure(2)
    plt.clf()
    plt.title(f'Results (n={n})')
    plt.xlabel('Step')
    plt.ylabel('Score')

    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    if path is not None:
        plt.savefig(fname=path)
    else:
        plt.show()
