from .base import Session, Unit
import tables as tb
import numpy as np
from ..utils import meta_loaders
from collections import defaultdict

STIM_POWER_FIELDNAME = 'pharosStimPower'
STIM_DURATION_FIELDNAME = 'pulseOnDur_1'


class TwoPUnit(Unit):
    """jon/gilad 2photon stimulation"""

    def template(self):
        pass

    def plot_template(self, *args, **kwargs):
        pass


class TwoPSession(Session):
    """jon/gilad axoclamp session skeleton."""
    unit_type = TwoPUnit

    def __init__(self, meta_path: str, spiketimes_path: str):
        """
        
        :param meta_path: path to meta h5 file. 
        :param spiketimes_path: path to spiketimes csv file.
        """
        fn_meta = meta_path
        self.subject_id, self.sess_id, self.rec_id = 1, 1, 1
        self._unit_subset = None

        self.paths = {
            'meta': fn_meta,
        }

        with tb.open_file(fn_meta, 'r') as f:  # type: tb.File
            try:
                self.fs = f.get_node_attr('/', 'acquisition_frequency_hz')
            except AttributeError:
                print('warning, no fs supplied in meta file. Using default of 25khz')
                self.fs = 20000.
            try:
                run_ends = f.get_node('/Events/run_ends').read()
                self.recording_length = run_ends.max()
            except tb.NoSuchNodeError:
                self.recording_length = None
            self.stimuli = self._make_stimuli(f)
        self._make_units(spiketimes_path)
        self._sniff = None  # container for sniff


    def _make_units(self, unit_path:str):
        spiketimes = np.fromfile(unit_path, sep=',')
        unit = self.unit_type(0, spiketimes, 5, self)
        self._unit_info = {'ids': (0,), 'ratings': (5,)}
        self._units = (unit, )
        return

    def _make_stimuli(self, meta_file: tb.File):
        events = meta_file.root.Events
        trials = meta_loaders.load_aligned_trials(meta_file)
        laser_events = events.laser.read()
        self.laserevents = laser_events
        self.laserstarts = laser_events[:,0]
        self.laserstarts_by_power = defaultdict(list)
        self.laserstarts_by_dur = defaultdict(list)

        allstarts = np.array([x[0] for x in trials])
        allrows = [x[1] for x in trials]
        n_trs = len(trials)

        for pulse in laser_events:
            p_st, p_nd = pulse
            p_tr_idx = np.searchsorted(allstarts, p_st) - 1  # returns the index of the first number >= value, we want the number before
            # Make sure that searchsorted is behaving as expected:
            if p_tr_idx > 0:
                tr_st, tr_row = trials[p_tr_idx]
                assert tr_st < p_st
            if p_tr_idx < n_trs -1:
                nxt_tr_st, _ = trials[p_tr_idx + 1]
                assert nxt_tr_st > p_st

            trialrow = allrows[p_tr_idx]
            power = float(trialrow[STIM_POWER_FIELDNAME])
            self.laserstarts_by_power[power].append(p_st)
            dur = trialrow[STIM_DURATION_FIELDNAME]
            self.laserstarts_by_dur[dur].append(p_st)
