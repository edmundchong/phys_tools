import numpy as np
from phys_tools.loaders import meta_loaders, spyking_loaders
import tables as tb
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.stats import norm


class Unit(ABC):
    """
    Container for unit information and related functions.
    """
    def __init__(self, unit_id, spiketimes, rating, session):
        """
        :param unit_id: Identifier for unit. typically int or string.
        :param spiketimes: array of spiketimes
        :param session: Session object
        """
        assert issubclass(type(session), Session)
        self.spiketimes = spiketimes
        self.session = session
        self.uid = unit_id
        self.rating = rating

    def get_epoch_samples(self, start: int, end: int) -> np.array:
        """
        Gets spikes falling between start and end times. Time is specified in samples.

        :param start: start sample
        :param end: end sample
        :return: array of spiketimes
        """
        if start >= end:
            raise ValueError('Start of epoch must fall before end.')
        mask = (self.spiketimes >= int(start)) & (self.spiketimes <= int(end))
        return self.spiketimes[mask]

    def get_epoch_ms(self, t_start, t_end) -> np.array:
        """
        Return spiketimes in millisecond time base.
        :param t_start: start time in milliseconds from recording start
        :param t_end: endtime
        :return: array of float times in millis.
        """

        st = int(self.session.millis_to_samples(t_start))
        nd = int(self.session.millis_to_samples(t_end))
        return self.session.samples_to_millis(self.get_epoch_samples(st, nd))

    def plot_psth_times(self, t_0s, pre_ms, post_ms, binsize_ms, axis=None, label='', color=None,
                        alpha=1., linewidth=2, linestyle='-', convolve=False):
        """
        Note: everything other than description of the zero time is in milliseconds. t_0 is described in samples!

        This function plots a PSTH using spike times from multiple epochs. The spikes to use are

        :param t_0: array of zero times (IN SAMPLES) for the PSTH
        :param pre_ms: number of ms to plot prior to t_0
        :param post_ms: number of ms to plot after t_0
        :param binsize_ms: binsize_ms (in ms)
        :param axis: matplotlib axis on which to plot (optional: otherwise, make new axis)
        :param label: string for plot legend if wanted
        :param color: matplotlib colorspec for the psth line (ie "k" for a black line)
        :param alpha: transparency of psth line (float: 1. is opaque, 0. is transparent)
        :param linewidth: line width for psth plot (float)
        :param linestyle: matplotlib linespec for psth plot
        :param convolve: default is false. If "gaussan" or "boxcar", use these shaped kernels to make plot instead of histogram.
        :return:
        """
        t_0s = np.asarray(t_0s)

        if (pre_ms + post_ms) % binsize_ms:
            d = (pre_ms + post_ms) % binsize_ms
            if d > binsize_ms / 2:  # round up or down depending on which side of the bin we're on:
                post_ms = post_ms + (binsize_ms - d)
            else:
                post_ms = post_ms - d
            print('warning the total window size is not divisible by the binsize_ms '
                  'adjusting the post time to {}.'.format(post_ms))

        binsize_samp = self.session.millis_to_samples(binsize_ms)
        pre_samp = int(self.session.millis_to_samples(pre_ms))
        post_samp = int(self.session.millis_to_samples(post_ms))
        n_samp = pre_samp + post_samp

        spiketrials, spiketimes, _ = self.get_rasters_samples(t_0s, pre_samp, post_samp)

        if not convolve:
            binsize = int(binsize_samp)
            nbins = int(n_samp / binsize_samp)
            bin_right_edges = [binsize * x - pre_samp for x in range(nbins+1)]
            psth, _ = np.histogram(spiketimes, bins=bin_right_edges)
            x = np.linspace(-pre_ms + .5 * binsize_ms, post_ms - .5 * binsize_ms, nbins)
            assert len(psth) == len(x)
        else:
            ss = np.zeros(n_samp)
            for i, j in enumerate(range(-pre_samp, post_samp)):
                ss[i] = (spiketimes == j).sum()
            if convolve == 'gaussian':
                kernel = norm.pdf(np.linspace(-3, 3, int(binsize_ms * self.session.fs/1000)))
            elif convolve == 'boxcar':
                kernel = np.ones(binsize_samp)
            else:
                raise ValueError("valid convolution parameters are 'gaussian' and 'boxcar'.")
            psth = np.convolve(ss, kernel, mode='valid')
            x = np.linspace(-pre_ms + .5 * binsize_ms, post_ms - .5 * binsize_ms, len(psth))

        # scale to Hz
        n_trials = len(t_0s)
        if convolve:
            sec_per_bin = kernel.sum() * n_trials / self.session.fs
        else:
            sec_per_bin = binsize_ms * n_trials / 1000
        psth_hz = psth / sec_per_bin
        # print (x)
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

    def get_rasters_ms(self, t_0s_ms, pre_ms, post_ms) -> (np.array, np.array, tuple):
        """
        Millisecond raster arrays from msec parameters. Returns tuple of 2 arrays and a size:

            trial numbers array, spiketimes (in ms) array, shape: (ntrials, (t_min, t_max))

        Times are all relative ot t_0 of each trial. Shape is required for subsequent plotting/rasterization.

        :param t_0s_ms: 1d array with times which will correspond with index zero in each row.
        :param pre_ms: number of milliseconds before t0s to return.
        :param post_ms: number of milliseconds after t0s to return.
        :return: trial array, spiketimes(ms) array, shape: (ntrials, (t_min, t_max))
        """
        nstarts = len(t_0s_ms)
        spiketimes = []
        spiketrials = []
        start_times_ms = t_0s_ms - pre_ms
        size = pre_ms + post_ms
        for i in range(nstarts):
            st = start_times_ms[i]
            t0 = t_0s_ms[i]
            spike_sub = self.get_epoch_ms(st, st + size) - t0  # everything here is float or at least signed.
            spiketrials.extend([i] * len(spike_sub))
            spiketimes.append(spike_sub)
        spiketimes = np.concatenate(spiketimes)
        spiketrials = np.array(spiketrials)
        shape = (nstarts, (-pre_ms, post_ms))
        return spiketrials, spiketimes, shape  # these are in milliseconds!!

    def get_rasters_samples(self, t_0s, pre, post) -> (np.array, np.array, tuple):
        """
        Sample timebase raster arrays from sample time parameters. Returns tuple of 2 arrays and a size:

            trial numbers array, spiketimes (in samples) array, shape: (ntrials, (t_min, t_max))

        Times are all relative ot t_0 of each trial. Shape is required for subsequent plotting/rasterization.

        :param t_0s: 1 d array with start reference times.
        :param pre: number of samples before t0 to return for each trial.
        :param post: number of samples after t0 to return for each trial.
        :return: trial array, spiketimes(samples) array, shape: (ntrials, (t_min, t_max))
        """
        spikes = self.spiketimes
        nstarts = len(t_0s)
        start_times = t_0s - pre
        size = pre + post
        spiketimes = []
        spiketrials = []
        start_times = start_times.astype(np.uint64)
        for i in range(nstarts):
            st = start_times[i]
            spike_sub = spikes[(spikes > st) & (spikes < st + size)]
            spike_sub -= st
            spike_sub = spike_sub.astype(np.int64) - pre  # uint can't be negative.
            spiketrials.extend([i]*len(spike_sub))
            spiketimes.append(spike_sub)
        spiketimes = np.concatenate(spiketimes)
        spiketrials = np.array(spiketrials)
        shape = (nstarts, (-pre, post))
        return spiketrials, spiketimes, shape

    def plot_rasters(self, rasters, x=None, axis=None, quick_plot=True, color=None, alpha=1, offset=0,
                     markersize=5):
        """
        Plots rasters gotten from get_rasters functions. Rasters are described as a matrix (ntrials, time).

        Timing is specified in a non-intuitive way: Unless an 'x' array is specified, each index of the raster arrays
        is considered the time for plotting. If x is specified, raster[j] will be plotted at the time specified
        by x[j].

        Obviously, rasters time dimension and x array must have the same length.

        :param rasters: matrix of rasters (ntrials, size)
        :param x: time axis. This is used to define the time for each index of the rasters. (ie raster[0] will be plotted at x[0])
        :param axis: matplotlib axis on which to plot (optional: otherwise, make new axis)
        :param quick_plot: if False, use a much slower rasterization method that draws lines instead of just using dots.
        :param color: matplotlib colorspec for psth line (ie 'k' for black line)
        :param alpha: transparency of psth line (float: 1. is opaque, 0. is transparent)
        :param offset: used for plotting muliple conditions on same axis.
        :param markersize: size of marker to use for plotting.
        :return:
        """

        if type(rasters) == tuple and len(rasters) == 3:
            trials, times, shape = rasters  # need ntrials because of sparse activity
            ntrials, (x_min, x_max) = shape
        else:
            trials, times = np.where(rasters)
            ntrials = len(rasters)
            x_min, x_max = None, None
            if x is not None:  # transform from matrix indices to time specified by x array.
                if not len(rasters.T) == len(x):
                    raise ValueError('raster array has time length of {}, x array has length of {}.'
                                     'These must be equal.'.format(len(rasters.T), len(x)))
                times[:] = x[times]
                x_min, x_max = x.min(), x.max()

        trials += offset + 1  # so we start at trial 1 on the plot and not trial 0.

        if axis is None:
            axis = plt.axes()

        if quick_plot:
            axis.scatter(times, trials, marker='.', c=color, alpha=alpha, s=markersize)
            plt.xlabel('Time')
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.spines['left'].set_visible(False)
            axis.xaxis.set_ticks_position('bottom')
            axis.set_yticks([1, ntrials])
        else:
            for tr, tr_ras in zip(trials, times):
                axis.vlines(tr_ras, tr + .5, tr + 1.5, color=color, alpha=alpha, linewidth=markersize)
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.xaxis.set_ticks_position('bottom')
                axis.set_yticks([])
        if x_min:
            axis.set_xlim([x_min, x_max])
        return axis

    def __str__(self):
        return "Unit {}".format(self.uid)


class Session(ABC):
    unit_type = Unit

    def __init__(self, dat_file_path, suffix='-1'):
        self.fn_dat = dat_file_path
        fn_templates, fn_results, fn_meta = spyking_loaders.make_file_paths(dat_file_path, suffix)
        self.filenames = {
            'templates': fn_templates,
            'results': fn_results,
            'meta': fn_meta
        }
        sk_units = spyking_loaders.load_spiketimes_unstructured(fn_templates, fn_results)
        self._make_units(sk_units)

        with tb.open_file(fn_meta, 'r') as f:
            try:
                self.fs = f.get_node_attr('/', 'acquisition_frequency_hz')
            except:
                print('warning, no fs supplied in meta file. Using default of 25khz')
                self.fs = 25000.
            self.stimuli = self._make_stimuli(f)


    def _make_units(self, unit_info, ):
        """
        different session types will instantiate different Unit types.
        """
        numbers, spiketimes, ratings = unit_info
        self._unit_info = {'ids': numbers, 'ratings': ratings}  #save this to use for quicker indexing.
        self.units = [self.unit_type(i, st, r, self) for i, st, r in zip(numbers, spiketimes, ratings)]
        return

    @abstractmethod
    def _make_stimuli(self, meta_file):
        """
        Class specific - extract the stimuli you are expecting from your file.
        :return:
        """
        pass

    def filter_units(self, ratings: list) -> list:

        """
        Filters units based ratings.
        :param ratings: list or range of ratings to include. 5 is A, 1 is F
        :return: list of units matching this criterion.
        """

        if type(ratings) == int:
            ratings = [ratings]
        mask = np.ones(len(self.units), dtype=bool)
        unit_ratings = self._unit_info['ratings']
        for r in ratings:
            mask &= unit_ratings == r
        if mask.any():
            w = np.where(mask)[0]
            units = [self.units[x] for x in w]
        else:
            units = []

        return units

    def filter_units_gte(self, rating: int) -> list:
        """
        Filters out units with ratings less than specified.\
        :param rating: integer of the lowest allowable rating. 5 is A, 1 is F
        :return: list of units with rating greater than or equal to rating.
        """
        unit_ratings = self._unit_info['ratings']
        mask = unit_ratings >= rating
        if mask.any():
            w = np.where(mask)[0]
            units = [self.units[x] for x in w]
        else:
            units = []
        return units

    def samples_to_millis(self, times):
        """
        convert sample time to milliseconds
        """
        factor = self.fs / 1000. # number of samples per millisecond from samp/sec
        return self._convert_times(times, factor)

    def millis_to_samples(self, samples):
        """
        convert times in ms
        :param samples:
        :return:
        """
        factor = 1000. / self.fs
        return self._convert_times(samples, factor)

    def _convert_times(self, times, factor):
        if type(times) == np.ndarray:
            return times / factor
        elif type(times) in (list, tuple):
            return [x / factor for x in times]
        elif np.isscalar(times):
            return times / factor
        else:
            raise ValueError('sorry, samples_to_millis cannot use this datatype.')
