import numpy as np
from phys_tools.utils import meta_loaders, spyking_loaders
import tables as tb
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.stats import norm
# from scipy.signal import fftconvolve
import numba as nb
import os
import time

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
        self.unit_id = unit_id
        self.rating = rating
        self._template = None  # to be loaded on demand by load_template().

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

    def get_psth_times(self, t_0s, pre_ms, post_ms, binsize_ms, convolve=False):
        """
        Returns an array PSTH.

        :param t_0: array of zero times (IN SAMPLES) for the PSTH
        :param pre_ms: number of ms to plot prior to t_0
        :param post_ms: number of ms to plot after t_0
        :param binsize_ms: binsize_ms (in ms)
        :return: (time_scale_ms, psth in hz)
        """

        t_0s = np.asarray(t_0s)
        if not convolve and (pre_ms + post_ms) % binsize_ms:
            d = (pre_ms + post_ms) % binsize_ms
            if d > binsize_ms / 2:  # round up or down depending on which side of the bin we're on:
                post_ms = post_ms + (binsize_ms - d)
            else:
                post_ms = post_ms - d
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
            ss = np.zeros(n_samp, dtype='float32')
            spiketimes_as_idx = spiketimes + pre_samp
            assert np.all(spiketimes_as_idx >= 0)
            for spk in spiketimes_as_idx:  # numba doesn't make this loop quicker under normal cases.
                ss[spk] += 1.
            if convolve == 'gaussian':
                kernel = norm.pdf(np.linspace(-3, 3, binsize_samp)).astype('float32')
            elif convolve == 'boxcar':
                kernel = np.ones(binsize_samp, dtype=np.float32)
            else:
                raise ValueError("valid convolution parameters are 'gaussian' and 'boxcar'.")
            psth = np.convolve(ss, kernel, mode='valid')
            x = np.linspace(-pre_ms + .5 * binsize_ms, post_ms - .5 * binsize_ms, len(psth))

        n_trials = len(t_0s)
        if convolve:
            sec_per_bin = kernel.sum() * n_trials / self.session.fs
        else:
            sec_per_bin = binsize_ms * n_trials / 1000
        psth_hz = psth / sec_per_bin

        return x, psth_hz

    def plot_psth_times(self, t_0s, pre_ms, post_ms, binsize_ms, axis=None, label='', color=None,
                        alpha=1., linewidth=2, linestyle='-', convolve=False, setaxislabels=True):
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

        x, psth_hz = self.get_psth_times(t_0s, pre_ms, post_ms, binsize_ms, convolve)
        if axis is None:
            axis = plt.axes()
        axis.plot(x, psth_hz, label=label, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        if setaxislabels:
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
        max_spikes = 10000
        spiketimes = np.zeros(max_spikes, dtype=np.int)
        spiketrials = np.zeros(max_spikes, dtype=np.int)
        spikes = self.spiketimes
        nstarts = len(t_0s)
        start_times = t_0s - pre
        size = pre + post
        start_times = start_times.astype(np.uint64)
        c = 0
        for i in range(nstarts):
            st = start_times[i]
            i_st, i_stop = np.searchsorted(spikes, [st, st+size])
            spike_sub = spikes[i_st:i_stop].copy()
            # spike_sub = spikes[(spikes > st) & (spikes < st + size)].copy()
            spike_sub -= st
            spike_sub = spike_sub.astype(np.int) - pre  # uint can't be negative.
            nspikes = len(spike_sub)
            spiketimes[c:c+nspikes] = spike_sub
            spiketrials[c:c+nspikes] = i
            c += nspikes
        shape = (nstarts, (-pre, post))
        return spiketrials[:c], spiketimes[:c], shape

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
        return "{}u{:03d}".format(self.session, self.unit_id)

    @property
    def template(self):
        """only loads template data if/when requested."""
        if self._template is None:
            self._template = spyking_loaders.load_templates(self.session.filenames['templates'], self.unit_id)
        return self._template

    def plot_template(self, x_scale=20, y_scale=2, axis=None, color='b',
                        alpha=1., linewidth=1, linestyle='-',):
        """

        :param x_scale:
        :param y_scale:
        :param axis:
        :param color:
        :param alpha:
        :param linewidth:
        :param linestyle:
        :return:
        """
        template = self.template
        probe_positions = self.session.probe_geometry
        if probe_positions is None:
            raise ValueError('Session {} is missing probe geometry. Check for prb file.')
        assert len(template) == len(probe_positions)
        if axis is None:
            axis = plt.axes()

        for i in range(len(template)):
            waveform = template[i]
            # if np.any(waveform):
            x_offset, y_offset = probe_positions[i]
            x = np.linspace(0, x_scale, len(waveform)) + x_offset
            y = waveform * y_scale + y_offset

            axis.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        return axis

    def plot_autocorrelation(self, binsize_ms=1, range_ms=30, axis=None, color='b'):
        """
        Plots autocorellogram for unit. Normalized to all spikes.


        :param binsize_ms:
        :param range_ms:
        :param axis:
        :return:
        """

        range_samp = self.session.millis_to_samples(range_ms)
        binsize_samp = self.session.millis_to_samples(binsize_ms)

        if range_samp% binsize_samp:
            range_samp -= range_samp % binsize_samp

        nbins = int(range_samp // binsize_samp)
        bin_edges_s = np.array([binsize_samp * (x+1) for x in range(nbins)], dtype='uint64')
        total_spikes = len(self.spiketimes)
        # seed_indexes = np.random.randint(0,  total_spikes - 1, min((20000, total_spikes // 10)))
        # seed_indexes = np.arange(0, total_spikes)
        bins = _autocorrelation(self.spiketimes, bin_edges_s,)
        bin_edges_ms = self.session.samples_to_millis(bin_edges_s) - 1.

        if axis is None:
            axis = plt.axes()

        bin_p = bins / total_spikes

        axis.plot(bin_edges_ms, bin_p, drawstyle='steps-post', fillstyle='bottom', color=color)
        axis.set_xlim(0, None)
        axis.set_ylim(0, None)
        return axis

# @nb.jit
# def _find_idxes(sorted_array, val, st=-1, nd=-1):
#     if st < 0 and nd < 0:
#         st = 0, nd = sorted_array
#     if sorted_array[st] > val and sorted_array[nd]

@nb.jit('int64[:](uint64[:], uint64[:])', cache=True)
def _autocorrelation(spiketimes, bin_edges, ):
    nbins = bin_edges.size
    bins = np.zeros(nbins, dtype='int64')
    range_samp = bin_edges.max()
    nspikes = spiketimes.size
    for i in range(nspikes):
        seed = spiketimes[i]
        j = i+1
        curr = spiketimes[j]
        dist = curr - seed
        while dist < range_samp and j < nspikes:
            # dist = curr - seed
            notyet = True
            k = 0
            while notyet and k < nbins:
                if dist < bin_edges[k]:
                    notyet = False
                    bins[k] += 1
                k += 1
            j += 1
            curr = spiketimes[j]
            dist = curr - seed
    return bins


class Session(ABC):
    unit_type = Unit

    def __init__(self, dat_file_path, suffix='-1'):
        self.fn_dat = dat_file_path
        self.subject_id, self.sess_id, self.rec_id = self._parse_path(dat_file_path)
        fn_templates, fn_results, fn_meta = spyking_loaders.make_file_paths(dat_file_path, suffix)
        fn_probe = spyking_loaders.find_probe_file(dat_file_path)
        self.filenames = {
            'templates': fn_templates,
            'results': fn_results,
            'meta': fn_meta,
            'probe': fn_probe
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
        if fn_probe:
            self.probe_geometry = spyking_loaders.load_probe_positions(fn_probe)
        else:
            self.probe_geometry = None
        self._sniff = None  # container for sniff property

    def __str__(self):
        return "m{}s{:02d}r{}".format(self.subject_id, self.sess_id, self.rec_id.upper())

    def _make_units(self, unit_info, ):
        """
        different session types will instantiate different Unit types.
        """
        numbers, spiketimes, ratings = unit_info
        self._unit_info = {'ids': numbers, 'ratings': ratings}  #save this to use for quicker indexing.
        self._units = [self.unit_type(i, st, r, self) for i, st, r in zip(numbers, spiketimes, ratings)]
        return

    @staticmethod
    def _parse_path(path: str) -> tuple:
        """
        Parse path structured like .../mouse_XXXX/sess_YYY/Z.dat
        Xs parsed as subject_id number, Ys are session number, Z is rec name

        :param path: path string to parse. May be relative or absolute.
        :return (subject_id, session, rec)
        """
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        a = path.find('mouse_')
        subject = int(path[a + 6:a + 10])
        c = path.find('sess_')
        sessionnumber = int(path[c + 5:c + 8])
        recname = path[c + 9:c + 10]
        return subject, sessionnumber, recname

    @abstractmethod
    def _make_stimuli(self, meta_file):
        """
        Class specific - extract the stimuli you are expecting from your file.
        :return:
        """
        pass

    def units(self, unit_ids=None, ratings=None) -> list:
        """
        Returns units and can filter units by id or unit ratings. Default behavior is to return all units in the
        session.

        :param unit_ids: list of units to return (optional: return all units if not specified)
        :param ratings: list or range of ratings to include. 5 is A, 1 is F (optional, return all units if specified)
        :return: list of units matching this criterion.
        """
        if not (unit_ids is None or ratings is None):
            raise ValueError('Only one filter can be applied: unit id or rating')
        mask = np.ones(len(self._units), dtype=bool)
        if unit_ids is not None:
            all_ids = self._unit_info['ids']
            if type(unit_ids) == int:
                unit_ids = [unit_ids]
            for u in unit_ids:
                mask &= all_ids == u
        elif ratings is not None:
            if type(ratings) == int:
                ratings = [ratings]
            unit_ratings = self._unit_info['ratings']
            for r in ratings:
                mask &= unit_ratings == r
        w = np.where(mask)[0]
        return [self._units[x] for x in w]

    def units_gte(self, rating: int) -> list:
        """
        Filters out units with ratings less than specified.\
        :param rating: integer of the lowest allowable rating. 5 is A, 1 is F
        :return: list of units with rating greater than or equal to rating.
        """
        unit_ratings = self._unit_info['ratings']
        mask = unit_ratings >= rating
        if mask.any():
            w = np.where(mask)[0]
            units = [self._units[x] for x in w]
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

    def get_first_odor_sniffs(self, odor: str, concentration):
        """
        returns the first inhalations and exhalations of specified odorant.

        :param odor: string specifying odor
        :param concentration: numeric specifying concentration of odor.
        :return: tuple (inhalations, exhalations) of arrays.
        """

        odors = self.stimuli['odors']
        concs = self.stimuli['odorconcs']
        inhales = self.stimuli['inhales']
        exhales = self.stimuli['exhales']
        odormask = odors == odor
        concmask = concs == concentration
        allmask = odormask & concmask
        idxes = np.where(allmask)[0]
        first_inhs, first_exhs = [], []
        for i in idxes:
            inhs, exhs = inhales[i], exhales[i]
            if len(inhs) and len(exhs):
                first_inhs.append(inhs[0])
                first_exhs.append(exhs[0])
        return np.array(first_inhs), np.array(first_exhs)

    @property
    def sniff(self) -> np.array:
        """
        loads all sniff samples from the session meta file.
        """
        if self._sniff is None:
            with tb.open_file(self.filenames['meta'], 'r') as f:
                self._sniff = meta_loaders.load_sniff_trace(f)
        return self._sniff

    def get_sniff_traces(self, t_0s, pre_ms, post_ms) -> np.ndarray:
        """
        Loads and returns sniff sample values around specified sniff t_0s.

        :param t_0s: array or list of t_0s.
        :param pre_ms: number of ms to return prior to specified t_0s.
        :param post_ms: number of ms to return after specified t_0s.
        :return: sniffs in 2d array (Nsniffs, Nsamples) (C-order)
        """

        pre_samps, post_samps = self.millis_to_samples((pre_ms, post_ms))

        if np.isscalar(t_0s):
            n_sniffs = 1
            t_0s = np.array([t_0s])
        else:
            n_sniffs = len(t_0s)
        sniff = self.sniff
        sniff_mat = np.zeros((n_sniffs, int(pre_samps + post_samps)), dtype=sniff.dtype)
        for i in range(n_sniffs):
            t = t_0s[i]
            st = int(t - pre_samps)
            nd = int(t + post_samps)
            sniff_mat[i, :] = sniff[st:nd]
        return sniff_mat

    def plot_sniffs(self, t_0s, pre_ms, post_ms, axis=None, color='b', alpha=1., linewidth=2, linestyle='-'):
        """
        Plots sniff trace around times specified by t_0s (specified in samples)

        :param t_0s: array or list of t_0s specified in *samples*
        :param pre_ms: number of ms to return prior to specified t_0s.
        :param post_ms: number of ms to return after specified t_0s.
        :param axis: existing matplotlib axis on which to plot. Default will create new axis.
        :param color: matplotlib colorspec for the psth line (ie "k" for a black line)
        :param alpha: transparency of psth line (float: 1. is opaque, 0. is transparent)
        :param linewidth: line width for psth plot (float)
        :param linestyle: matplotlib linespec for psth plot
        :return:
        """

        sniffs = self.get_sniff_traces(t_0s, pre_ms, post_ms)
        x = np.linspace(-pre_ms, post_ms, num=len(sniffs.T))
        if axis is None:
            axis = plt.axes()
        for i in range(len(sniffs)):
            axis.plot(x, sniffs[i, :], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        axis.plot([0] * 2, plt.ylim(), '--k', linewidth=1)
        return axis
