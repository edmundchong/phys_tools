from scipy.signal import medfilt, decimate, butter, filtfilt, firwin
from numba import jit
import numpy as np
try:
    import matplotlib.pyplot as plt
except RuntimeError:
    pass
import tables as tb
from tqdm import tqdm


def _preprocess(sniff, decimation_factor=25):
    """
    Decimate and median filter sniff.

    :param sniff: sniff stream array
    :param decimation_factor: factor by which to decimate. a good value gets you to 1kHz
    :return: sniff array.
    """
    s_ds = decimate(sniff, decimation_factor, zero_phase=True)  # decimate to 1 khz
    s_ds -= s_ds.mean()
    s_ds_med = medfilt(s_ds, 41)
    return s_ds_med


@jit
def _findmin(vals, start):
    d = -1
    i = start
    while d <= 0:
        d = vals[i-1] - vals[i]
        i -= 1
    return i


def _find_threshold(sniff_filtered, forced_max=None, debug=False):
    ns, b = np.histogram(sniff_filtered, bins=300)
    if forced_max is None:
        m = np.where(ns == ns.max())[0][0]
    else:
        m = (np.abs(b - forced_max)).argmin()
        # m = forced_max
    ind = _findmin(ns, m)
    th = b[ind]

    if debug:
        plt.plot(b[:-1], ns, '*-', label='hist')
        plt.plot([b[m]] * 2, plt.ylim(), label='max')
        plt.plot([th] * 2, plt.ylim(), label="threshold: {0:0.2f}".format(th))
        plt.title('Threshold finder debug plot.')
        plt.ylabel('mv')
        plt.xlabel('number')
        plt.legend(loc='best')
        plt.show()
    return th


def _threshold_xings(array, threshold):
    """
    finds upward _downward_ crossings of threshold for an array
    :param array: array of values (ie sniff).
    :param threshold: threshold value.
    :return: threshold crossings
    """
    sniff_log = array < threshold
    inhaling = sniff_log[0]
    last_i = 0
    inhales = np.zeros(200000, dtype=np.int)
    n = 0
    for i in range(len(sniff_log)):
        val = sniff_log[i]
        if not inhaling and val and i-last_i > 15:
            inhaling = True
            last_i = val
            inhales[n] = i
            n += 1
        elif inhaling and not val:
            inhaling = False

    return inhales[:n]


# @jit
def nested_inh(sniff, downsample=25, debug=True):
    """
    Uses a combination of thresholding the derivative to determine the onset of inhalation. Threshold is used
    to define a window around which to search for the maximum

    :param sniff:
    :param downsample: factor to downsample (usually trying to get the fs to something around 1kHz).
    :param debug: Plots threshold graph.
    :return:
    """
    s_ds_med = _preprocess(sniff, downsample)
    threshold = _find_threshold(s_ds_med, debug=debug)
    i1 = _threshold_xings(s_ds_med, threshold)
    i2 = np.zeros_like(i1)
    for i in range(len(i1)):
        st = i1[i] - 30
        if st < 0:
            st = 0
        nd = i1[i]
        snip = s_ds_med[st:nd]
        d_snip = np.diff(snip)
        maxder = np.where(d_snip == d_snip.min())[0][-1] + st
        i2[i] = maxder
    i2 *= downsample
    return i2


def proof_inhales(sniff, onsets, n_plots=10, winsize=400, downsample_factor=25):
    """
    :param sniff: sniff stream array.
    :param onsets: onset times to check. Downsampling will occur automatically.
    :param n_plots: number of random inhales to choose to plot.
    :param winsize: size of plotting window. inhale is centered within the window.
    :param downsample_factor: factor to downsample for plotting, etc.
    :return: none
    """

    sniff = _preprocess(sniff, decimation_factor=downsample_factor)
    nr = onsets / downsample_factor
    for i in np.random.randint(0, len(nr), n_plots):
        st = nr[i] - int(winsize/2)
        nd = nr[i] + int(winsize/2)
        print (st)
        print(nd)
        plt.plot(sniff[st:nd])
#         plt.plot(sniff_filt[st:nd])
        plt.plot([int(winsize/2)] * 2, plt.ylim())
#         plt.plot(plt.xlim(), [th]*2)
        plt.plot(plt.xlim(), [0]*2)
        plt.show()
    return


# @jit
def find_offset_der(sniff_filtered, onsets, threshold):
    """
    Finds sniff ends using a combination of threshold and derivative. Uses threshold crossings (negative to
    positive) to define a search window, then looks for a derivative condition to be met to define sniff off-
    set.

    :param sniff_filtered: Sniff (downsampled and filtered).
    :param onsets: sniff onsets
    :param threshold:
    :return:
    """

    offs = np.zeros_like(onsets)
    #     print (len(onsets))
    max_idx = len(sniff_filtered) - 1
    for i in range(len(onsets)):
        on = int(onsets[i])
        threshold_cross_on = False
        threshold_cross_off = False
        j = on
        #         print(on)
        #         print(j)
        # find the off threshold crossing index.

        stop = np.min([on + 300, max_idx])
        while not threshold_cross_off and j < stop:
            j = j + 1
            samp = sniff_filtered[j]
            if not threshold_cross_on and samp < threshold:
                threshold_cross_on = True
            elif threshold_cross_on and samp > threshold:
                threshold_cross_off = True
        # extract the next few samples and find the inflection point:
        if j < max_idx:
            snip = sniff_filtered[j:j + 200]
            d = np.diff(snip)
            for ii in range(len(d) - 1):
                if d[ii + 1] < d[ii]:
                    off = ii + j
                    #                 print(j)
                    #                 print (off)
                    offs[i] = off
                    break
                if ii == len(snip) - 3:
                    print('wtf')
    return offs

def find_offset_2ndder(sniff_filtered, onsets, threshold):
    """
    Finds sniff ends using a combination of threshold and derivative. Uses threshold crossings (negative to
    positive) to define a search window, then looks for a derivative condition to be met to define sniff off-
    set.

    :param sniff_filtered: Sniff (downsampled and filtered).
    :param onsets: sniff onsets
    :param threshold:
    :return:
    """

    offs = np.zeros_like(onsets)
    #     print (len(onsets))
    max_idx = len(sniff_filtered) - 1
    for i in range(len(onsets)):
        on = int(onsets[i])
        threshold_cross_on = False
        threshold_cross_off = False
        j = on
        #         print(on)
        #         print(j)
        # find the off threshold crossing index.

        stop = np.min([on + 300, max_idx])
        while not threshold_cross_off and j < stop:
            j = j + 1
            samp = sniff_filtered[j]
            if not threshold_cross_on and samp < threshold:
                threshold_cross_on = True
            elif threshold_cross_on and samp > threshold:
                threshold_cross_off = True
        # extract the next few samples and find the inflection point:
        if j < max_idx:
            snip = sniff_filtered[j:j + 200]
            d = np.diff(snip)
            for ii in range(len(d) - 1):
                if d[ii + 1] < d[ii]:
                    off = ii + j
                    #                 print(j)
                    #                 print (off)
                    break
                if ii == len(snip) - 3:
                    print('wtf')
            d2 = np.diff(d)
            off2 = np.where(d2[ii:ii+20] == np.min(d2[ii:ii+20]))[0][0]+j
            offs[i] = off2
    return offs

# @jit
def find_inh_exhs(sniff, decimation_factor=25, debug=True, threshold_force_max=None, debug_winsize=400):
    """
    Finds the inhalations and exhalations in sniff stream.

    Decimates and filters the sniff, finds a threshold automatically through a semi-robust way. Then,
    using the thresholds to define search windows, we're looking for changes in the derivative

    :param sniff: raw sniff array.
    :param decimation_factor: decimation factor to get to 1 kHz.
    :param debug: Plot debug plots.
    :param threshold_force_max:
    :param debug_winsize:
    :return: inhalations, exhalations
    """

    s_ds_med = _preprocess(sniff, decimation_factor)
    threshold = _find_threshold(s_ds_med, forced_max=threshold_force_max, debug=debug)
    i1 = _threshold_xings(s_ds_med, threshold)
    inhales = np.zeros_like(i1)
    for i in range(len(i1)):
        st = i1[i] - 30
        if st < 0:
            st = 0
        nd = i1[i]
        snip = s_ds_med[st:nd]
        d_snip = np.diff(snip)
        maxder = np.where(d_snip == d_snip.min())[0][-1] + st
        inhales[i] = maxder

    exhales = find_offset_der(s_ds_med, inhales, threshold)

    if debug:
        proof_inh_exh(s_ds_med, inhales, exhales, winsize=debug_winsize)

    inhales *= decimation_factor  # return to original sniff coordinates.
    exhales *= decimation_factor

    return inhales, exhales

def find_inh_exhs2(sniff, decimation_factor=25, debug=True, threshold_force_max=None, debug_winsize=400):
    """
    Finds the inhalations and exhalations in sniff stream.

    Adds butterworth filter to the above

    Decimates and filters the sniff, finds a threshold automatically through a semi-robust way. Then,
    using the thresholds to define search windows, we're looking for changes in the derivative

    :param sniff: raw sniff array.
    :param decimation_factor: decimation factor to get to 1 kHz.
    :param debug: Plot debug plots.
    :param threshold_force_max:
    :param debug_winsize:
    :return: inhalations, exhalations
    """

    s_ds_med = _preprocess(sniff, decimation_factor)
    # b, a = butter(2, .15, 'lowpass')
    h = firwin(12, .1)
    s_ds_med_buttered = filtfilt(h, 1., s_ds_med)

    threshold = _find_threshold(s_ds_med_buttered, forced_max=threshold_force_max, debug=debug)
    i1 = _threshold_xings(s_ds_med_buttered, threshold)
    inhales = np.zeros_like(i1)
    for i in tqdm(range(len(i1)), unit='sniffs', desc='Inhale detection'):
        st = i1[i] - 30
        if st < 0:
            st = 0
        nd = i1[i]
        snip = s_ds_med_buttered[st:nd]
        d_snip = np.diff(snip)
        maxder = np.where(d_snip == d_snip.min())[0][-1] + st
        inhales[i] = maxder

    exhales = find_offset_2ndder(s_ds_med_buttered, inhales, threshold)

    if debug:
        proof_inh_exh(s_ds_med_buttered, inhales, exhales, winsize=debug_winsize)
        _inh_vals = np.zeros(len(inhales), dtype=s_ds_med_buttered.dtype)
        _exh_vals = np.zeros(len(exhales), dtype=s_ds_med_buttered.dtype)
        for i in range(len(inhales)):
            _inh_vals[i] = s_ds_med_buttered[inhales[i]]
        for i in range(len(exhales)):
            _exh_vals[i] = s_ds_med_buttered[exhales[i]]
        plt.subplot(1,2,1)
        plt.hist(_inh_vals, bins=100)
        plt.subplot(1,2,2)
        plt.hist(_exh_vals, bins=100)
        plt.show()

    inhales *= decimation_factor  # return to original sniff coordinates.
    exhales *= decimation_factor

    return inhales, exhales



def proof_inh_exh(sniff, inhs, exhs, n_plots=10, winsize=400):
    """
    Plots random inhalations and exhalations for proofreading/debugging purposes.

    No decimating or time stamp scaling is done here, so everything must be in the same time base.

    :param sniff: sniff stream
    :param inhs: array of inhalation times (should be on same time base as the sniff array)
    :param exhs: array of exhalations times (should be on same time base as the sniff array)
    :param n_plots: number of random inh-exhs to plot.
    :param winsize: size of plotting window. inhale is centered within the window.
    :return:
    """
    x = np.arange(-winsize/2, winsize/2)
    for i in np.random.randint(0, len(inhs), n_plots):
        inn = inhs[i]
        exx = exhs[i]
        st = inn - int(winsize/2)
        nd = inn + int(winsize/2)
        plt.plot(x, sniff[st:nd])
        d = exx - inn
        inh_plot = 0
        plt.plot([inh_plot] * 2, plt.ylim())
        plt.plot([inh_plot + d] * 2, plt.ylim())
        plt.grid()
        plt.show()
    return


def load_sniff_stream(meta_h5_path):
    """
    Loads sniff from meta file created with this package

    :param meta_h5_path: path to the meta h5 file.
    :return: raw sniff array.
    """

    with tb.open_file(meta_h5_path, 'r') as f:
        sniff = f.root.Streams.sniff.read()
    return sniff


def save_sniff_times(meta_h5_path, inhalations, exhalations, suffix=1):
    """
    Saves inhalations and exhalation arrays to the h5 meta file created with this package.

    :param meta_h5_path: path to hdf5 meta file.
    :param inhalations: array of inhalation onset times
    :param exhalations: array of exhalation onset (ie inhalation offset times)
    :return:
    """

    with tb.open_file(meta_h5_path, 'r+') as f:
        f.create_carray('/Events', 'inhalations_{}'.format(suffix), obj=inhalations, createparents=True)
        f.create_carray('/Events', 'exhalations_{}'.format(suffix), obj=exhalations, createparents=True)
    return



