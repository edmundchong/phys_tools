from .base import Session, Unit
from ..utils import meta_loaders
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ODOR_FIELD = "olfas:olfa_{}:odor"
ODOR_CONC_FIELD = "odorconc"
TRIAL_NUM_FIELD = "trialNumber"
INHALATION_NODE = "/Events/inhalations_{}"
EXHALATION_NODE = "/Events/exhalations_{}"


class OdorUnit(Unit):
    def __init__(self, unit_id, spiketimes: np.ndarray, rating, session):
        super(OdorUnit, self).__init__(unit_id, spiketimes, rating, session)

    def get_odor_psth(self, odor, concentration, pre_ms, post_ms, binsize_ms, convolve=False):
        """
        Returns an array PSTH.

        :param odor: odor name (str)
        :param concentration:
        :param pre_ms: number of milliseconds prior to inhalation to plot
        :param post_ms: number of milliseconds after inhalation to plot
        :param binsize_ms: binsize for histogram
        :param convolve: default is false. If "gaussan" or "boxcar", use these shaped kernels to make plot instead of histogram.
        :return: Psth
        """
        inhs, exhs = self.session.get_first_odor_sniffs(odor, concentration)
        return self.get_psth_times(inhs, pre_ms, post_ms, binsize_ms, convolve)

    def plot_odor_psth(self, odor, concentration, pre_ms, post_ms, binsize_ms,
                       axis=None, label='', color=None, alpha=1., linewidth=2, linestyle='-',
                       convolve=False):
        """
        Plots odor PSTHs for first inhalations of a specified odorant at a specified concentration.
        Wraps the BaseUnit.plot_psth_times function.

        :param odor: odor name (str)
        :param concentration:
        :param pre_ms: number of milliseconds prior to inhalation to plot
        :param post_ms: number of milliseconds after inhalation to plot
        :param binsize_ms: binsize for histogram
        :param axis: matplotlib axis on which to plot (optional: otherwise, make new axis)
        :param label: string for plot legend
        :param color: matplotlib colorspec for psth line (ie 'k' for black line)
        :param alpha: transparency of psth line (float: 1. is opaque, 0. is transparent)
        :param linewidth: line width for psth plot (float)
        :param linestyle: matplotlib linespec for psth plot
        :param convolve: default is false. If "gaussan" or "boxcar", use these shaped kernels to make plot instead of histogram.
        :return: axis
        """

        inhs, exhs = self.session.get_first_odor_sniffs(odor, concentration)
        return self.plot_psth_times(inhs, pre_ms, post_ms, binsize_ms, axis=axis, label=label, color=color,
                                    alpha=alpha, linewidth=linewidth, linestyle=linestyle, convolve=convolve)

    def get_odor_rasters(self, odor, concentration, pre_ms, post_ms):
        """

        :param odor:
        :param concentration:
        :param pre_ms:
        :param post_ms:
        :return:
        """
        inhs, exhs = self.session.get_first_odor_sniffs(odor, concentration)
        t_0s_ms = self.session.samples_to_millis(inhs)
        return self.get_rasters_ms(t_0s_ms, pre_ms, post_ms)

    def plot_odor_rasters(self, odor, concentration, pre_ms, post_ms, sniff_overlay=False, axis=None,
                          quick_plot=True, color=None, alpha=1, offset=0, markersize=.5):
        """
        Plots rasters for given odor and concentration condition.

        offset parameter can be used for plotting multiple conditions on the same axis. For instance, if
        condition_1 has 10 trials,call the function for condition_1 with offset of 0, and for condition_2 use
        an offset of 10.

        :param odor: odor name (str)
        :param concentration: odor concentration to plot.
        :param pre_ms: number of milliseconds prior to inhalation to plot
        :param post_ms: number of milliseconds after inhalation to plot
        :param sniff_overlay: sorts rasters by inhalation length and overlays a polygon representing the period of first inhalation.
        :param axis: matplotlib axis on which to plot (optional: otherwise, make new axis)
        :param quick_plot: if False, use a much slower rasterization method that draws lines instead of just using dots.
        :param color: matplotlib colorspec for psth line (ie 'k' for black line)
        :param alpha: transparency of psth line (float: 1. is opaque, 0. is transparent)
        :param offset: used for plotting muliple conditions on same axis.
        :param markersize: size of marker to use for plotting.
        :return: plot axis
        """
        inhs, exhs = self.session.get_first_odor_sniffs(odor, concentration)
        inhs_ms, exhs_ms = map(self.session.samples_to_millis, (inhs, exhs))

        if sniff_overlay:

            n_tr = len(inhs)
            diffs_samp = exhs - inhs
            diffs_ms = self.session.samples_to_millis(diffs_samp)
            order = np.argsort(diffs_ms)
            inhs_ms, exhs_ms = [x[order] for x in (inhs_ms, exhs_ms)]
            diffs_ms_ordered = diffs_ms[order]
            points = [(0, 1)]  # start at trial 1, not 0 consistent with how we're plotting the rasters.
            points.extend([(diffs_ms_ordered[x], x + 1) for x in range(n_tr)])
            points.append((0, n_tr))
            poly = Polygon(points, color='g', alpha=.25)
            # TODO: polygon is having trouble covering the points of trial 1. not sure why.
        rasters = self.get_rasters_ms(inhs_ms, pre_ms, post_ms)
        ax = self.plot_rasters(rasters, axis=axis, quick_plot=quick_plot, color=color, alpha=alpha,
                               offset=offset, markersize=markersize)
        if sniff_overlay:
            ax.add_patch(poly)
        return ax

    def plot_odor_rasters_warp(self, odor, concentration, pre_ms, post_ms, sort_sniffs=True, axis=None,
                               quick_plot=True, color=None, alpha=1, offset=0, markersize=5):
        """
        Plots odor response rasters in time that is warped to duration of first inhalation.

        Warping is of the ith spike in the jth trial:
            spiketime[i, j] *= ( µ_inhale_duration / inhalation_duration[j] )

        :param odor: odor name (str)
        :param concentration: odor concentration to plot.
        :param pre_ms: number of milliseconds prior to inhalation to plot
        :param post_ms: number of milliseconds after inhalation to plot
        :param sort_sniffs: if True (default), sort the sniffs by the inhalation length.
        :param axis: matplotlib axis on which to plot (optional: otherwise, make new axis)
        :param quick_plot: if False, use a much slower rasterization method that draws lines instead of just using dots.
        :param color: matplotlib colorspec for psth line (ie 'k' for black line)
        :param alpha: transparency of psth line (float: 1. is opaque, 0. is transparent)
        :param offset: used for plotting muliple conditions on same axis.
        :param markersize: size of marker to use for plotting.
        :return: plot axis
        """
        inhs, exhs = self.session.get_first_odor_sniffs(odor, concentration)
        ntrials = len(inhs)
        diffs = exhs - inhs
        if sort_sniffs:
            # sort inhalation onsets, exhalations and diffs by inhalation length. Rasters will be ordered
            sort_indexes = np.argsort(diffs)
            inhs, exhs, diffs = [x[sort_indexes] for x in (inhs, exhs, diffs)]
        scalars = diffs.mean() / diffs
        pre, post = self.session.millis_to_samples((pre_ms, post_ms))
        allspikes = []
        alltrials = []
        for i in range(ntrials):
            t_0 = inhs[i] # + self.session.millis_to_samples(50.)
            st = t_0 - pre  # constant.
            s = scalars[i]
            spikes = (self.get_epoch_samples(st, t_0 + post / s) - st).astype(
                float)  # subtract start before converting from uint to not lose precision
            spikes -= pre  # subtract to get negative pre-t0 times AFTER conversion to float because uint will break
            spikes[spikes > 0] *= s  # scaling floats is right thing to do here.
            allspikes.append(spikes)
            alltrials.extend([i] * len(spikes))  # start at trial '1'
        allspikes = np.concatenate(allspikes)
        alltrials = np.array(alltrials)
        allspikes_ms = self.session.samples_to_millis(allspikes)
        shape = ntrials, (-pre_ms, post_ms)
        return self.plot_rasters((alltrials, allspikes_ms, shape), axis=axis, quick_plot=quick_plot,
                                 color=color, alpha=alpha, offset=offset, markersize=markersize)


class OdorSession(Session):
    unit_type = OdorUnit

    def __init__(self, meta_fn):
        super(OdorSession, self).__init__(meta_fn)
        self._odors = None
        self._unique_concs = None
        self._concentrations_by_odor = None

    def _make_stimuli(self, meta_file: tb.File) -> dict:
        """

        :param meta_f: tb.File object.
        :return: stimulus dictionary
        """
        events = meta_file.root.Events
        trials = meta_loaders.load_aligned_trials(meta_file)
        fv = events.finalvalve.read()
        fv_starts = fv[:, 0]
        fv_ends = fv[:, 1]
        n_trs = len(trials)
        self.inhales, self.exhales = meta_loaders.load_sniff_events(meta_file)

        # building lists of stimulus attributes. Doing this because masking is easy/efficient with np arrays.
        odors_by_stim = []
        inhales_by_stim = []
        exhales_by_stim = []
        finalvalve_on_times = []
        finalvalve_off_times = []
        concentrations_by_stim = []

        _odor_fields_valid = []
        for jj in range(4):  # check for olfa nodes for subsequent extractions.
            s = ODOR_FIELD.format(jj)
            if s in trials[0][1].dtype.names:
                _odor_fields_valid.append(s)
        # todo: get odor concentrations from multiple olfactometers!!!!
        for i_tr in range(n_trs - 1):
            nd = trials[i_tr + 1][0]
            st, tr = trials[i_tr]
            odors_by_olfa = []
            for s in _odor_fields_valid:
                odor = tr[s]
                if odor:
                    odors_by_olfa.append(odor.decode())  # make into string from binary.
            assert len(odors_by_olfa) < 2, 'multiple concentrations functionality is not included.'
            if len(odors_by_olfa):
                odor = odors_by_olfa[0]
            else:
                odor = 'blank'
            cs = tr[ODOR_CONC_FIELD]
            fv_starts_trial_mask = (fv_starts > st) & (fv_starts < nd)
            if fv_starts_trial_mask.any():
                # if there is no final valve opening, we don't want to add anything to the stim dictionary.
                if fv_starts_trial_mask.sum() > 1:
                    print('warning, there are more than 1 final valve openings in trial.')
                # we're taking only the first fvon, any additional are due to trial numbering problems.
                fvon = fv_starts[fv_starts_trial_mask][0]
                fvoff = fv_ends[fv_ends > fvon][0]  # first end following start is the end.
                # process inhalations and exhalations falling within the stimulus time period.
                inhs_fv = np.array([])
                exhs_fv = np.array([])
                inhale_mask_fv = ((self.inhales > fvon) & (self.inhales < fvoff))
                if inhale_mask_fv.any():
                    inhs_fv = self.inhales[inhale_mask_fv]
                    inh_idxes = np.where(inhale_mask_fv)[0]
                    first_inh, last_inh = inhs_fv.min(), inhs_fv.max()
                    num_inhs = len(inh_idxes)
                    exh_mask = self.exhales > first_inh
                    if exh_mask.any():
                        first_exh_idx = np.where(exh_mask)[0][0]
                        last_exh_idx = min((first_exh_idx + num_inhs + 1, len(self.exhales)))
                        exhs_fv = self.exhales[first_exh_idx:last_exh_idx]  # n inhales == n exhales

                # these appends only happen if FV opening is detected:
                inhales_by_stim.append(inhs_fv)
                exhales_by_stim.append(exhs_fv)
                finalvalve_on_times.append(fvon)
                finalvalve_off_times.append(fvoff)
                odors_by_stim.append(odor)
                concentrations_by_stim.append(cs)

        result = {
            'fv_ons': np.array(finalvalve_on_times),
            'fv_offs': np.array(finalvalve_off_times),
            'odors': np.array(odors_by_stim),
            'odorconcs': np.array(concentrations_by_stim),
            'inhales': np.array(inhales_by_stim),
            'exhales': np.array(exhales_by_stim)
        }
        return result

    @property
    def odors(self) -> np.array:
        """
        Returns all odors found within the session.
        """
        if self._odors is None:
            self._odors = np.unique(self.stimuli['odors'])
        return self._odors

    def get_concentrations(self, odor: str) -> np.array:
        """
        Returns a sorted array of unique concentrations presented for a specified odorant.
        """
        odors = self.stimuli['odors']
        odormask = odors == odor
        concs = self.stimuli['odorconcs']
        return np.unique(concs[odormask])

    @property
    def concentrations_by_odor(self) -> dict:
        if self._concentrations_by_odor is None:
            self._concentrations_by_odor = {}
            for o in self.odors:
                self._concentrations_by_odor[o] = self.get_concentrations(o)
        return self._concentrations_by_odor

    def plot_odor_sniffs(self, odor: str, conc, pre_ms, post_ms, axis=None, separate_plots=False, color='b', alpha=1.,
                         linewidth=2, linestyle='-', ):
        """
        Plots sniffs around the first inhalation of odor.

        :param odor: odor specification
        :param conc: odor concentration specification
        :param pre_ms: number of ms to return prior to specified t_0s.
        :param post_ms: number of ms to return after specified t_0s.
        :param axis: existing axis on which to plot. Default will create new plot axis.
        :param separate_plots: if True, plot each sniff on separate axis with inhalation and exhalation marked.
        :param color: matplotlib colorspec for the psth line (ie "k" for a black line)
        :param alpha: transparency of psth line (float: 1. is opaque, 0. is transparent)
        :param linewidth: line width for psth plot (float)
        :param linestyle: matplotlib linespec for psth plot
        :return:
        """

        inhs, exhs = self.get_first_odor_sniffs(odor, conc)
        if separate_plots:
            for i in range(len(inhs)):
                self.plot_sniffs(inhs[i], pre_ms, post_ms, color=color, alpha=alpha, linestyle=linestyle,
                                 linewidth=linewidth)
                plt.plot([self.samples_to_millis(exhs[i] - inhs[i])] * 2, plt.ylim())
                plt.show()
        else:
            axis = self.plot_sniffs(inhs, pre_ms, post_ms, axis=axis, color=color, alpha=alpha, linestyle=linestyle,
                             linewidth=linewidth)  # will create/return new plot if None is supplied
        return axis

