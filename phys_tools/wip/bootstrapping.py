import os

import numpy as np
from numba import jit

from phys_tools.wip.basic_plotting import get_odor_sniffs

k_width_ms = 30
k_width_samp = int(k_width_ms)
kernel = (np.ones(k_width_samp)/k_width_samp)

@jit
def _bt(r, n_tr):
    a = np.random.randint(0, r.shape[0], n_tr)
    b = r[a, :]
    c = (b.sum(axis=0) / n_tr)
    return c

@jit
def _draw(r, n_tr, kernel, n_bt) -> np.ndarray:

    n_samps = r.shape[1]
    bootstrap_results = np.zeros((n_bt, n_samps))
    for i in range(n_bt):
        c = _bt(r, n_tr)
        d = np.convolve(kernel, c, 'same')
        bootstrap_results[i, :] = d
    return bootstrap_results


def draw_distributions(r_null, r_stim, kernel):
    n_bt = 3000
    n_tr = r_stim.shape[0]
    bootstrap_null = _draw(r_null, n_tr, kernel, n_bt)
    bootstrap_stim = _draw(r_stim, n_tr, kernel, n_bt)
    bootstrap_stim.sort(axis=0)
    bootstrap_null.sort(axis=0)
    return bootstrap_null, bootstrap_stim


def draw_and_save(units, prefix, inhs_null, inhs_stim):
    """

    :param units: dictionary of units' spiketimes
    :param prefix: prefix for the savefiles
    :param inhs_null: array of inhalation times for null distribution
    :param inhs_stim: aray of inhalation times for stimulus distribution
    :return:
    """
    if inhs_null:
        for k, unit in units.items():
            r = ephys_tools2.wip.basic_plotting.get_rasters(inhs_null, unit, decimate_factor=25)
            r_stim = ephys_tools2.wip.basic_plotting.get_rasters(inhs_stim, unit, decimate_factor=25)

            bootstrap_null, bootstrap_stim = draw_distributions(r, r_stim, kernel, )

            np.save('{}_u{}_null.npy'.format(prefix, k), bootstrap_null)
            np.save('{}_u{}_stim.npy'.format(prefix, k), bootstrap_stim)
    return


def bootstrap_all_concs(rec_name, rec_stim, rec_units, rec_noodorinhs):

    concs = np.unique(rec_stim['odorconcs'])
    odors = np.unique(rec_stim['odors'])

    d = '/workspace/_bootstrap/{}'.format(rec_name)
    os.mkdir(d)

    for odor in odors:
        odormask = rec_stim['odors'] == odor
        for c in concs:
            if np.any(rec_stim['odorconcs'][odormask] == c):
                prefix = '{}/{}:{:0.2e}'.format(d, odor, c)
                stim_sniffs = get_odor_sniffs(odor, c, rec_stim)[0]
                if len(stim_sniffs):
                    draw_and_save(rec_units, prefix, rec_noodorinhs, stim_sniffs)


def bootstrap_ex_1(rec_stim, rec_units, rec_noodorinhs, significance):
    """

    :param rec_stim:
    :param rec_units:
    :param rec_noodorinhs:
    :param significance:  degree of overlap between stim and null distributions that is considered a response.
    :return: excitation latencies by unit, inhibition latencies by unit.
    """
    concs = np.unique(rec_stim['odorconcs'])
    odors = np.unique(rec_stim['odors'])

    ex_latencies_by_unit = {}
    in_latencies_by_unit = {}

    for uname, unit in rec_units.items():
        r_null = ephys_tools2.wip.basic_plotting.get_rasters(rec_noodorinhs, unit, decimate_factor=25)
        ex_latencies_by_odor = dict()
        in_latencies_by_odor = dict()
        for odor in odors:
            ex_latencies_by_odor[odor] = {}
            in_latencies_by_odor[odor] = {}
            odormask = rec_stim['odors'] == odor
            for c in concs:
                if np.any(rec_stim['odorconcs'][odormask] == c):
                    stim_sniffs = get_odor_sniffs(odor, c, rec_stim)[0]
                    r_stim = ephys_tools2.wip.basic_plotting.get_rasters(stim_sniffs, unit, decimate_factor=25)
                    bootstrap_null, bootstrap_stim = draw_distributions(r_null, r_stim, kernel, )
                    p_in = _inhib_p(bootstrap_stim, bootstrap_null, .05)
                    p_ex = _excite_p(bootstrap_stim, bootstrap_null, .95)

                    for d, p in zip((in_latencies_by_odor, ex_latencies_by_odor), (p_in, p_ex)):

                        if np.any(p < significance):

                            d[odor][c] = np.where(p < significance)[0][0]
                        else:
                            d[odor][c] = -1
        in_latencies_by_unit[uname] = in_latencies_by_odor
        ex_latencies_by_unit[uname] = ex_latencies_by_odor
    return ex_latencies_by_unit, in_latencies_by_unit


def bootstrap_ex_2(rec_stim, rec_units, rec_noodorinhs, significance, factor, sniff_filter=None,
                   ff_threshold=None):
    """

    :param rec_stim:
    :param rec_units:
    :param rec_noodorinhs:
    :param significance:  degree of overlap between stim and null distributions that is considered a response.
    :param factor: mean response must be at least this much bigger (or smaller for inhibition) to be considered
    significant.
    :return: excitation latencies by unit, inhibition latencies by unit.
    """
    concs = np.unique(rec_stim['odorconcs'])
    odors = np.unique(rec_stim['odors'])

    ex_latencies_by_unit = {}
    in_latencies_by_unit = {}

    for uname, unit in rec_units.items():
        r_null = ephys_tools2.wip.basic_plotting.get_rasters(rec_noodorinhs, unit, decimate_factor=25)
        ex_latencies_by_odor = dict()
        in_latencies_by_odor = dict()
        for odor in odors:
            ex_latencies_by_odor[odor] = {}
            in_latencies_by_odor[odor] = {}
            odormask = rec_stim['odors'] == odor
            for conc in concs:
                if np.any(rec_stim['odorconcs'][odormask] == conc):
                    stim_sniffs = get_odor_sniffs(odor, conc, rec_stim, filter=sniff_filter)[0]
                    r_stim = ephys_tools2.wip.basic_plotting.get_rasters(stim_sniffs, unit, decimate_factor=25)
                    bootstrap_null, bootstrap_stim = draw_distributions(r_null, r_stim, kernel, )
                    p_in = _inhib_p(bootstrap_stim, bootstrap_null, .05)
                    p_ex = _excite_p(bootstrap_stim, bootstrap_null, .95)
                    mean_stim = np.mean(bootstrap_stim, axis=0)
                    mean_null = np.mean(bootstrap_null, axis=0)

                    if ff_threshold:
                        var_stim = bootstrap_stim.var(axis=0)
                        ff = var_stim/mean_stim
                        ff_thresholded = ff <= ff_threshold
                    else:
                        ff_thresholded = np.ones(len(mean_stim), dtype=np.bool)
                    d = ex_latencies_by_odor
                    p = p_ex
                    a = p < significance
                    b = (mean_stim / mean_null) > factor
                    c = mean_stim > .01
                    final = a & b & c & ff_thresholded
                    if np.any(final):
                        d[odor][conc] = np.where(final)[0][0]
                    else:
                        d[odor][conc] = -1

                    d = in_latencies_by_odor
                    p = p_in
                    a = p < significance
                    b = (mean_null/mean_stim) < factor
                    final = a * b
                    if np.any(final):
                        d[odor][conc] = np.where(final)[0][0]
                    else:
                        d[odor][conc] = -1

        in_latencies_by_unit[uname] = in_latencies_by_odor
        ex_latencies_by_unit[uname] = ex_latencies_by_odor
    return ex_latencies_by_unit, in_latencies_by_unit




@jit
def _inhib_p(array1, array2, percentile, start=k_width_samp // 2, ):
    """
    find the number of _bootstrap draws of the stimulus that are HIGHER than the minimum of the null.

    :param array1: stim array
    :param array2: null array
    :param percentile: at what percentage should you take the "minimum" of the null. "0" would be the
    actual minimum value found in the null.
    :param start: where to start making probabilities used to get rid of edge effects from convolution.
    :return:
    """
    array1.sort(axis=0)
    array2.sort(axis=0)
    nbt, ns = array1.shape
    ps = np.ones(ns)
    min_i = _percentile_to_index(nbt, percentile)
    minima = array2[min_i, :]

    for i in range(start, ns):
        a1 = array1[:, i]
        n = np.sum(a1 > minima[i])
        ps[i] = n / nbt
    return ps


@jit
def _excite_p(array1, array2, percentile, start=k_width_samp // 2):
    nbt, ns = array1.shape
    ps = np.ones(ns)
    max_i = _percentile_to_index(nbt, percentile)
    maxima = array2[max_i, :]

    for i in range(start, ns):
        a1 = array1[:, i]
        n = np.sum(a1 < maxima[i])
        ps[i] = n / nbt
    return ps


@jit
def _percentile_to_index(array_length, percentile):
    """
    This gives the index of the value at the given percentile of a SORTED ARRAY (with small values at low
    indices).

    :param array_length: number of values in the array
    :param percentile:
    :return: int
    """
    a = percentile * array_length
    b = np.rint(a)
    return int(b)