from .base import *
from ..utils import meta_loaders
import numpy as np


SPOT_FIELD_NAME = 'spots'
SPOTSIZE_FIELD_NAME = 'spotsizes'
INTENSITY_FIELD_NAME = 'LaserIntensity_MWmm2'  # assuming homogenous intensity for all spots.


class PatternUnit(Unit):
    """Unit from PatternSession"""

    def __init__(self, unit_id, spiketimes, rating, session):
        super(PatternUnit, self).__init__(unit_id, spiketimes, rating, session)

        # TODO: make some good plotting functions.

    def plot_spots(self, pre_ms, post_ms, binsize_ms, sequences=None, axis=None, label='', color=None,
                   alpha=.8, linewidth=2, linestyle='-', convolve='gaussian'):
        """
        Plots all the psths for all spots.

        :param pre_ms: number of ms to plot prior to t_0
        :param post_ms: number of ms to plot after t_0
        :param binsize_ms: binsize_ms (in ms)
        :param axis: matplotlib axis on which to plot (optional: otherwise, make new axis)
        :param label: string for plot legend if wanted
        :param color: matplotlib colorspec for the psth line (ie "k" for a black line)
        :param alpha: transparency of psth line (float: 1. is opaque, 0. is transparent)
        :param linewidth: line width for psth plot (float)
        :param linestyle: matplotlib linespec for psth plot
        :param convolve: default is false. If "gaussian" or "boxcar", convolve spikes w/ kernels with these
        shapes to make plot instead of histogram.
        :return:
        """
        offset_x = (pre_ms + post_ms) * 1.05
        offset_y = 200.
        xs = []
        ys = []

        if axis is None:
            axis = plt.axes()  #type: Axes

        if not sequences:
            sequences = self.session.sequence_dict.keys()
        for seq in sequences:
            times = self.session.sequence_dict[seq]
            if len(seq) == 1 and len(seq.frames) == 1:
                spot = seq.frames[0].spots[0]
                p_x = spot.x
                p_y = spot.y - 3  #TODO THIS IS A STUPID HACK TO CORRECT OFFSET!!!!!!
                x, psth = self.get_psth_times(times, pre_ms, post_ms, binsize_ms, convolve=convolve)
                o_y = -offset_y * p_y
                o_x = offset_x * p_x
                psth += o_y
                x += o_x
                ys.append(psth)
                xs.append(x)
                mx, mn = x.max(), x.min()
                axis.plot([mx, mn], [o_y] * 2, color='k', linewidth=.75)
                axis.plot([o_x]*2, [o_y, o_y+100], color='k', linewidth=.75)
        ys = np.asarray(ys)
        xs = np.asarray(xs)
        axis.plot(xs.T, ys.T, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)


class PatternSession(Session):
    """Session with Polygon pattern stimuli."""

    unit_type = PatternUnit

    def __init__(self, *args, **kwargs):
        self.sequence_dict = dict()
        self.unique_frames = []
        self.unique_spots = set()
        self.sequences = []
        self._intensities = set()
        self._coordinates = set()
        super(PatternSession, self).__init__(*args, **kwargs)
        with tb.open_file(self.paths['meta'], 'r') as f:
            self.inhales, self.exhales = meta_loaders.load_sniff_events(f)

    def _make_stimuli(self, meta_file):
        """
        Extract stimuli from meta file.
        """
        events = meta_file.root.Events
        trials = meta_loaders.load_aligned_trials(meta_file)
        laser_events = events.laser.read()
        allstarts = [x[0] for x in trials]
        n_trs = len(trials)
        pulse_starts = laser_events[:, 0]
        sequence_dict = {}

        for i in range(n_trs - 1):
            st, tr = trials[i]
            spots_str = tr[SPOT_FIELD_NAME].decode()  # spots are stored as a list of x,y coordinates (ie [[2,3], [5,3]])
            spot_size_str = tr[SPOTSIZE_FIELD_NAME].decode()  # decode transforms bytes to str
            if spots_str:
                laser_spots = meta_loaders.poly_spot_to_list(spots_str)
                spot_sizes = meta_loaders.poly_spot_to_list(spot_size_str)
                spot_intensity = tr[INTENSITY_FIELD_NAME]  # type: float
                nd = allstarts[i + 1]
                pulse_idxes = np.where((pulse_starts >= st) & (pulse_starts < nd))[0]
                numpulses = len(pulse_idxes)
                sequence_frames = list()
                sequence_times = list()
                if numpulses - 2 == len(laser_spots):
                    # assert numpulses - 2 == len(laser_spots)
                    assert meta_loaders.find_list_depth(laser_spots) < 4
                    # minus 2 for two blank frame pulses, one before and one after the stimulus spots.
                    # even if we are presenting multiple spots in a frame (or pulse), they will be
                    # represented in a list. So we'll have a spot lists like: [[[2,3],[4,3]], [[2,1], [3,4]]],
                    # where we have 2 frames each with 2 spots, but the length of laser_spots is 2.
                    for j in range(1, numpulses - 1):  # were going to ignore the first and last stim spots (see above).
                        i_p = pulse_idxes[j]
                        i_frame = j - 1  # the spots start at index 0
                        t_frame = pulse_starts[i_p]
                        spots_frame = laser_spots[i_frame]
                        spot_size = spot_sizes[i_frame]
                        depth = meta_loaders.find_list_depth(spots_frame)
                        if depth == 1:  # only one spot in frame ([[3,3], [4,4]])
                            sf = Spot(spots_frame, spot_size, spot_intensity)  # making tuple because needs to be hashable.
                            self.unique_spots.add(sf)
                            # spots are represented as [y, x]
                            frame = Frame((sf,))
                        elif depth == 2:
                            spots_frame.sort()  # sort so that we guarantee consistent ordering of spot lists with same spots
                            sfs = tuple([Spot(x, spot_size, spot_intensity) for x in spots_frame])  # make everything tupled.
                            self.unique_spots.add(*sfs)
                            frame = Frame(sfs)
                        else:
                            raise ValueError('problem with spot list depth.')
                        sequence_frames.append(frame)
                        sequence_times.append(t_frame)
                    sequence = FrameSequence(tuple(sequence_frames), tuple(sequence_times))  #TODO: utils an put frametimes relative in this.
                    sequence_start_time = sequence_times[0]
                    if sequence in sequence_dict.keys():
                        sequence_dict[sequence].append(sequence_start_time)
                    else:
                        sequence_dict[sequence] = [sequence_start_time]
                    self.sequences.append(sequence)
                else:
                    print('warning {} frames found in trial {} and {} pulses found. Discarding trial.'.format(
                        len(laser_spots), tr['trialNumber'], numpulses
                    ))
        self.sequence_dict = sequence_dict

    @property
    def unique_intensities(self):
        if not self._intensities:
            for s in self.unique_spots:  # type: Spot
                self._intensities.add(s.intensity)
        return self._intensities

    @property
    def unique_coordinates(self):
        if not self._coordinates:
            for s in self.unique_spots:  # type: Spot
                self._coordinates.add((s.x, s.y))
        return self._coordinates

    def filter_spots(self, coordinate=None, intensity=None):
        """
        Finds frame sequences where the coordinate and intensities specified are matched
        If parameters are not specified, the filter is not applied for that feature.
        :param coordinate: optional coordinate to match 
        :param intensity: 
        :return: 
        """
        coordinate_set = set()
        intensity_set = set()
        sequences = self.sequence_dict.keys()

        if coordinate is None:
            for f in sequences:
                coordinate_set.add(f)
        else:
            for f in sequences:  #type: FrameSequence
                if f.coordinates[0][0] == coordinate:
                    coordinate_set.add(f)

        if intensity is None:
            for f in sequences:
                intensity_set.add(f)
        else:
            for f in sequences:  # type: FrameSequence
                if f.intensities[0][0] == intensity:
                    intensity_set.add(f)
        return intensity_set & coordinate_set


class FrameSequence:
    """
    Sequence of Frames. Attributes: frames, frametimes, frametimes_relative.
    """
    def __init__(self, frames: tuple, frametimes: tuple, frametimes_relative=None):
        """
        Sequence of frames.

        :param frames: tuple of Frame models.
        :param frametimes: recording time where each frame is presented.
        :param frametimes_relative: This should be the timing of the frames relative to
        inhalation onset. This is used to determine whether the sequence is unique or an instance of an
        existing frame.
        """
        #TODO: need to allow for input of relative (or sniff relative) times, probably from Voyeur data.
        if not len(frames) == len(frametimes):
            raise ValueError('number of frames must be consistent with number of frametimes')
        if not type(frames) == tuple == type(frametimes):
            raise ValueError('must use tuples (hashablility)')
        assert all([type(x) == Frame for x in frames])
        # assert all([type(x) == int for x in frametimes])
        self.frames = frames  # type: list[Frame]
        self.frametimes = frametimes
        self.nframes = len(frames)
        self.frametimes_relative = frametimes_relative  # this is for
        self.start = frametimes[0]

    @property
    def intensities(self):
        return [x.intensities for x in self.frames]

    @property
    def coordinates(self):
        return [x.coordinates for x in self.frames]

    def __hash__(self):
        # we're using the relative frametimes here because this is
        return hash((self.frames, self.frametimes_relative))

    def __eq__(self, other):
        return len(self.frames) == len(other.frames) and self.frametimes_relative == other.frametimes_relative

    def __contains__(self, item):
        return self.frames.__contains__(item)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]

    def __iter__(self):
        return self.frames.__iter__()

    def __str__(self):
        return "FrameSequence"


class Frame:
    """
    Frames are sparsely defined boolean matrices that were projected onto the brain using a DMD.
    """
    def __init__(self, spots: tuple):
        """
        Frames are made up of spots. For now, spots are simply tuples expressing cartesian coordinates
        of the spot (x, y).

        A frame can contain as many spots as needed, again as a tuple.
        ie ((x1, y1), (x2, y2), ..., (xN, yN))

        (tuples is required because they are hashable and immutable)

        :param spots: tuple of spot coordinate tuples.
        """
        assert type(spots) == tuple
        self.spots = spots  # type: list[Spot]
        # todo store matrix dimensions for use making compact representation.

    @property
    def intensities(self):
        return [x.intensity for x in self.spots]

    @property
    def coordinates(self):
        return [(x.x, x.y) for x in self.spots]

    def __hash__(self):
        return hash(self.spots)

    def __eq__(self, other):
        return self.spots == other.spots

    def compact_representation(self):
        #  todo:  return frame as bool matrix
        pass

    def __len__(self):
        return len(self.spots)

    def __iter__(self):
        return self.spots.__iter__()

    def __getitem__(self, i):
        return self.spots[i]


class Spot:
    """basic unit of pattern stim ephys frame,"""

    def __init__(self, coordinates: tuple, size: float, intensity: float):
        """
        :param coordinates: tuple of
        :param size:
        :param intensity: laser intensity in mW mm-2. Taken from LaserIntensity_1 row.
        """
        self.y, self.x = coordinates  # note that this is flipped from sanity.
        # smaller y is anterior
        # smaller x is to the left
        self.size = size
        self.intensity = intensity

    def __hash__(self):
        return hash((self.x, self.y, self.size, self.intensity))

    def __str__(self):
        return "(x{}y{}s{}i{})".format(self.x, self.y, self.size, self.intensity)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.size == other.size and \
               self.intensity == other.intensity
