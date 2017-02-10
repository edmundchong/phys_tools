import numpy as np
from numba import jit
try:

    import matplotlib.pyplot as plt
except RuntimeError:
    pass

import seaborn as sns
import scipy as sp
# from scipy import signal
sns.set_style('ticks')

def get_responses(start_times, spikes, size=10000):
    """
    Returns a list of spike times following the start times.


    :param start_times:
    :param spikes:
    :param size:
    :return:
    """
    responses = list()
    for t in start_times:
        spike_sub = spikes[(spikes > t) & (spikes < t + size)] - t
        responses.append(spike_sub)
    return responses

@jit
def get_rasters(start_times, spikes, size=10000, decimate_factor=0):
    """

    :param start_times: array of times at which to make the rasters.
    :param spikes: the spiketimes of the unit.
    :param size: number of samples to consider for each raster.
    :return:
    """
    nstarts = len(start_times)

    if decimate_factor:
        rasters = np.zeros((nstarts, int(np.ceil(size/decimate_factor))+1), dtype=np.bool)
    else:
        rasters = np.zeros((nstarts, size+1), dtype=np.bool)

    for i in range(len(start_times)):
        t = int(start_times[i])
        spike_sub = spikes[(spikes > t) & (spikes < t + size)] - t

        if decimate_factor:
            dtype = spike_sub.dtype
            for ii in range(len(spike_sub)):
                spike_sub[ii] = np.rint(spike_sub[ii] / decimate_factor).astype(dtype)
        for ii in range(len(spike_sub)):
            s = int(spike_sub[ii])
            rasters[i, s] = True
    return rasters

# sns.swarmplot()

@jit
def decimate_timings(input_array, decimate_factor):
    """
    decimates in place. integers only please.

    :param input_array:
    :param decimate_factor:
    :return:
    """

    pass



# @jit
def get_rasters_warp1(start_times, spikes, size, scales,  decimate_factor=1):
    """

    :param start_times:
    :param spikes:
    :param size: size (in
    :param scales: scales are the factor by which to warp each trial.
    :param decimate_factor:
    :return:
    """
    nstarts = len(start_times)
    if decimate_factor:
        rasters = np.zeros((nstarts, int(np.ceil(size/decimate_factor))+1), dtype=np.bool)
    else:
        rasters = np.zeros((nstarts, size+1), dtype=np.bool)
    sz = rasters.shape[1]
    for i in range(len(start_times)):
        f = scales[i]
        t = int(start_times[int(i)])
        spike_sub = spikes[(spikes > t) & (spikes < t + size*scales[i])] - t
        spikes_warped = np.zeros_like(spike_sub)

        for ii in range(len(spike_sub)):
            spk = spike_sub[ii]
            spikes_warped[ii] = int(np.rint(spk * f / decimate_factor))
        for ii in range(len(spikes_warped)):
            s = spikes_warped[ii]
            if s < sz:
                rasters[i, s] = True
    return rasters

@jit
def _bootstrap(draw_matrix, raster_matrix, result_array, binsize_samp, nboots=1000):
    """
    used for plot_psth_raster.

    :param draw_matrix:
    :param raster_matrix:
    :param result_array:
    :param binsize_samp:
    :param nboots:
    :return:
    """
    for i in range(nboots):
        draw_i = draw_matrix[i,:]
        result_array[i, :] = make_psth(raster_matrix[draw_i,:], binsize_samp)
    return

@jit
def make_psth(rasters, binsize):
    """

    :param rasters: sparse arrays
    :param binsize:
    :return:
    """
    binsize = int(binsize)
    psth = rasters.sum(axis=0)
    nbins = int(np.floor(len(psth) / binsize))
    psth_binned = np.zeros(nbins, dtype=np.int)
    for i in range(nbins):
        binstart = i * binsize
        binstop = (i + 1) * binsize
        psth_binned[i] = np.sum(psth[binstart:binstop])
    return psth_binned


def plot_rasters(rasters, x=None, axis=None, quick_plot=True, colors=None, alpha=1, offset=0,
                 markersize=.5):
    """

    :param rasters:
    :param x:
    :param axis:
    :param quick_plot:
    :param colors:
    :param alpha:
    :param offset:
    :return:
    """
    # make event rasters:

    trials, times = np.where(rasters)
    if x is not None:
        for i in range(len(times)):
            times[i] = x[times[i]]
    ntrials = len(rasters)
    trials += offset + 1

    if axis is None:
        axis = plt.axes()

    if quick_plot:
        axis.scatter(times, trials, marker='.', c=colors, alpha=alpha, s=markersize)
        #         axis.relim()
        #         axis.autoscale_view(True,True,True)
        #         axis.set_ylim(-1, ntrials+1+offset)
        #         axis.set_xlim(left=0)
        # plt.ylabel('Trial')
        plt.xlabel('Time')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.xaxis.set_ticks_position('bottom')
        axis.set_yticks([1, ntrials])
    else:
        for tr, tr_ras in zip(trials, times):
            axis.vlines(tr_ras, tr + .5, tr + 1.5, color=colors, alpha=alpha, linewidth=markersize)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.spines['left'].set_visible(False)
            axis.xaxis.set_ticks_position('bottom')
            axis.set_yticks([])

    return axis



def plot_psth(spiketimes, t_0s, pre_ms, post_ms, binsize_ms, axis=None, fs=25000, label='', color=None,
               alpha=1., linewidth=2, linestyle='-', convolve=False):
    """
    Note: everything other than description of the zero time is in milliseconds. t_0 is described in samples!

    This function plots a PSTH using spike times from multiple epochs. The spikes to use are


    :param spiketimes: spiketimes for unit
    :param t_0: array of zero times (IN SAMPLES) for the PSTH
    :param pre_ms: number of ms to plot prior to t_0
    :param post_ms: number of ms to plot after t_0
    :param binsize_ms: binsize_ms (in ms)
    :param axis: matplotlib axis on which to plot (optional: otherwise, make new axis)
    :param fs: sampling frequency (Hz) (default is 25000 - 25 kHz)
    :param label: string for plot legend if wanted
    :param color: matplotlib colorspec for the psth line (ie "k" for a black line)
    :param alpha: transparency of psth line (float: 1. is opaque, 0. is transparent)
    :param linewidth: line width for psth plot (float)
    :param linestyle: matplotlib linespec for psth plot
    :param convolve: default is false. If "gaussan" or "boxcar", use these shaped kernels to make plot instead of histogram.
    :return:
    """
    t_0s = np.asarray(t_0s)

    if (pre_ms+post_ms) % binsize_ms:
        d = (pre_ms+post_ms) % binsize_ms
        if d > binsize_ms/2:  # round up or down depending on which side of the bin we're on:
            post_ms = post_ms + (binsize_ms - d)
        else:
            post_ms = post_ms - d
        print('warning the total window size is not divisible by the binsize_ms '
              'adjusting the post time to {}.'.format(post_ms))

    _s = int(fs/1000)  # conversion factor for ms to samples
    binsize_samp = binsize_ms * _s
    st = t_0s - pre_ms * _s
    n_samp = (post_ms + pre_ms) * _s

    rasters = get_rasters(st, spiketimes, n_samp)
    if not convolve:
        psth = make_psth(rasters, binsize_samp)
        # x = np.arange(-pre_ms+binsize_ms, post_ms, binsize_ms)
        nbins = (pre_ms + post_ms) / binsize_ms
        x = np.linspace(-pre_ms + .5 * binsize_ms, post_ms - .5 * binsize_ms, nbins)
        assert len(psth) == len(x)
    else:
        ss = rasters.sum(axis=0).astype(np.float)
        if convolve == 'gaussian':
            kernel = sp.stats.norm.pdf(np.linspace(-3, 3, 30*25))
        elif convolve == 'boxcar':
            kernel = np.ones(binsize_samp)
        else:
            raise ValueError("valid convolution parameters are 'gaussian' and 'boxcar'.")
        psth = np.convolve(ss, kernel, mode='valid')
        # psth = signal.fftconvolve(kernel, ss, 'valid')
        x = np. linspace(-pre_ms + .5 * binsize_ms, post_ms - .5 * binsize_ms, len(psth))


    #scale to Hz
    n_trials = len(t_0s)
    if convolve:
        sec_per_bin = kernel.sum() * n_trials / fs
    else:
        sec_per_bin = binsize_ms * n_trials / 1000
    psth_hz = psth/sec_per_bin

    if not axis:
        axis = plt.axes()
    axis.plot(x, psth_hz, label=label, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
    axis.set_ylabel('Firing rate (Hz)')
    axis.set_xlabel('Time (ms)')

    if axis.set_ylim()[1] < psth_hz.max():
        axis.set_ylim((0, psth_hz.max()))
    else:
        axis.set_ylim((0, None))
    return axis


def plot_psth_bootstrap(spiketimes, t_0s, pre_ms, post_ms, binsize_ms, axis=None, fs=25000, label='',
                        drawsize=None):
    """
    plots psth with 3-97 percentile error bars (~2sigma)

    Note: everything other than description of the zero time is in milliseconds. t_0 is described in samples!


    :param spiketimes: spiketimes for unit
    :param t_0: array of zero times (IN SAMPLES) for the PSTH
    :param pre_ms: number of ms to plot prior to t_0
    :param post_ms: number of ms to plot after t_0
    :param binsize_ms: binsize_ms (in ms)
    :param axis: matplotlib axis on which to plot (optional: otherwise, make new axis)
    :param fs: sampling frequency (Hz)
    :return:
    """

    nboots = 1000
    t_0s = np.asarray(t_0s)

    if (pre_ms+post_ms) % binsize_ms:
        d = (pre_ms+post_ms) % binsize_ms
        if d > binsize_ms/2:  # round up or down depending on which side of the bin we're on:
            post_ms = post_ms + (binsize_ms - d)
        else:
            post_ms = post_ms - d
        print('warning the total window size is not divisible by the binsize_ms '
              'adjusting the post time to {}.'.format(post_ms))

    _s = int(fs/1000)  # conversion factor for ms to samples
    binsize_samp = binsize_ms * _s
    st = t_0s - pre_ms * _s
    n_samp = (post_ms + pre_ms) * _s
    nbins = pre_ms+post_ms / binsize_ms
    print(pre_ms)
    print(post_ms)
    print(nbins)

    # x = np.arange(-pre_ms + binsize_ms, post_ms, binsize_ms)
    x = np.linspace(-pre_ms + .5*binsize_ms, post_ms - .5*binsize_ms, nbins)
    psths = np.zeros((nboots, len(x)))
    rasters = get_rasters(st, spiketimes, n_samp)

    if drawsize is None:
        drawsize = len(rasters)

    draws = np.random.randint(0, len(rasters) - 1, (nboots, drawsize))

    _bootstrap(draws, rasters, psths, binsize_samp, nboots)

    psth_high = np.percentile(psths, 97, axis=0)
    psth_low = np.percentile(psths, 3, axis=0)
    psth = np.percentile(psths, 50, axis=0)
    assert len(psth) == len(x)

    #scale to Hz
    n_trials = drawsize
    sec_per_bin = binsize_ms * n_trials / 1000
    psth_hz = psth / sec_per_bin
    psth_high_hz = psth_high / sec_per_bin
    psth_low_hz = psth_low / sec_per_bin


    if not axis:
        axis = plt
    pp = axis.plot(x, psth_hz, label=label, )
    axis.fill_between(x, psth_high_hz, psth_low_hz, alpha=.3, color=pp[0].get_color())
    axis.ylabel('Firing rate (Hz)')
    axis.xlabel('Time (ms)')

    if axis.ylim()[1] < psth_hz.max():
        axis.ylim((0, psth_hz.max()))
    else:
        axis.ylim((0,None))
    return axis


def plot_psth_warp1(spiketimes, inh_starts, inh_ends, scale_to, pre_ms, post_ms, binsize_ms,
                    axis=None, fs=25000, label='', color=None, alpha=1.):
    """

    :param spiketimes: unit spiketimes
    :param inh_starts: starts of inhalations to plot
    :param inh_ends: ends of inhalations to plot (used for scaling)
    :param scale_to: length of inhalations to scale to.
    :param pre_ms: ms prior to the inhalation with which to plot
    :param post_ms: ms after inh_starts to plot.
    :param binsize_ms: binsize for plotting.
    :param axis: axis on which to plot
    :param fs: sampling frequency (default 25 khz).
    :return:
    """

    t_0s = np.asarray(inh_starts)
    inh_lens = inh_ends - inh_starts
    scales = scale_to / inh_lens
    # print (scales)

    if (pre_ms+post_ms) % binsize_ms:
        d = (pre_ms+post_ms) % binsize_ms
        if d > binsize_ms/2:  # round up or down depending on which side of the bin we're on:
            post_ms = post_ms + (binsize_ms - d)
        else:
            post_ms = post_ms - d
        print('warning the total window size is not divisible by the binsize_ms '
              'adjusting the post time to {}.'.format(post_ms))

    _s = int(fs/1000)  # conversion factor for ms to samples
    binsize_samp = binsize_ms * _s
    st = t_0s - pre_ms * _s
    n_samp = (post_ms + pre_ms) * _s

    rasters = get_rasters_warp1(st, spiketimes, n_samp, scales)
    psth = make_psth(rasters, binsize_samp)
    x = np.arange(-pre_ms+binsize_ms, post_ms, binsize_ms)
    assert len(psth) == len(x)

    #scale to Hz
    n_trials = len(t_0s)
    sec_per_bin = binsize_ms * n_trials / 1000
    psth_hz = psth/sec_per_bin


    if not axis:
        axis = plt
    axis.plot(x, psth_hz, label=label, color=color, alpha=alpha)
    axis.ylabel('Firing rate (Hz)')
    axis.xlabel('Time (ms)')

    if axis.ylim()[1] < psth_hz.max():
        axis.ylim((0, psth_hz.max()))
    else:
        axis.ylim((0,None))
    return axis


@jit
def get_no_odor_sniffs(inh_times, exh_times, fv_ons, fv_offs, pad=30000, filter=None):
    """
    returns a tuple of arrays with times of inhalations and exhalations that do not occur within a final valve
    open time.

    :param inh_times:
    :param exh_times:
    :param fv_ons:
    :param fv_offs:
    :param pad: the
    :return:
    """

    out_of_odor_sniffs = np.zeros(len(inh_times), dtype=np.bool)
    padded_offs = fv_offs + pad
    for i in range(len(inh_times)):
        in_odor = False
        inh = inh_times[i]
        ii = 0
        l = len(fv_ons)
        while not in_odor and ii < l:
            on = fv_ons[ii]
            off = padded_offs[ii]
            if inh > on and inh < off:  # keep separate to short circuit the comparisons!
                in_odor = True
            ii += 1
        if not in_odor:
            out_of_odor_sniffs[i] = True
    d = exh_times - inh_times
    if filter is not None:
        filter_mask = d > filter[0]
        filter_mask &= d < filter[1]
        o = out_of_odor_sniffs & filter_mask
    else:
        o = out_of_odor_sniffs

    return inh_times[o], exh_times[o]


def get_odor_sniffs(odor, conc, stims, filter=None):
    """

    :param odor: odor to match
    :param conc: concentration to match
    :param stims:
    :return:
    """
    # indexes = np.zeros(len(inh_times), dtype=np.int)
    odormask = stims['odors'] == odor
    concmask = stims['odorconcs'] == conc
    allmask = odormask & concmask

    inhs = stims['inhales'][allmask]
    exhs = stims['exhales'][allmask]
    first_inhs = []
    first_exhs = []
    for x, y in zip(inhs, exhs):
        if len(x) and len(y):
            d = y[0] - x[0]
            if filter and d > filter[0] and d < filter[1]:
                first_inhs.append(x[0])
                first_exhs.append(y[0])
            elif filter is None:
                first_inhs.append(x[0])
                first_exhs.append(y[0])

    return np.array(first_inhs), np.array(first_exhs)
