from numba import jit
import numpy as np


# @jit(nopython=True)
# def _running_variance(array, windowsize=2500):
#     n_samples = len(array)
#     N = np.float(windowsize)
#     average = array[:windowsize].mean()
#     variance = array[:windowsize].var()
#     result = np.zeros(n_samples - windowsize)
#     result[0] = variance
#
#     for i in range(0, n_samples - windowsize):
#         new = array[windowsize+i]
#         old = array[i]
#         newaverage = average + new/N - old/N
#         variance += (new - old) * (new - newaverage + old - average) / (N - 1.)
#         result[i] = variance
#         average = newaverage
#     return result


@jit(nopython=True)
def _running_std(inn, windowsize=2500):
    array = inn.astype(np.float64)
    n_samples = len(array)

    n_bins = int(n_samples / windowsize)
    result = np.zeros(n_bins)

    for i in range(n_bins):
        st = i * windowsize
        nd = (i + 1) * windowsize
        result[i] = array[st:nd].var()
    return np.sqrt(result)


def matrix_running_std(matrix, windowsize=2500):
    """

    :param matrix:
    :param windowsize:
    :param nchs:
    :return:
    """

    n_samp, n_ch = matrix.shape
    n_bins = int(n_samp / windowsize)
    mean_std = np.zeros(n_bins)

    N = float(n_ch)

    for i in range(n_ch):  #calc running mean to save mem.
        s = _running_std(matrix[:, i], windowsize)
        mean_std += s / N

    return mean_std
