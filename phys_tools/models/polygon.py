from .base import *
from ..utils import meta_loaders
import numpy as np
from typing import Tuple


SPOT_FIELD_NAME = 'spots'
SPOTSIZE_FIELD_NAME = 'spotsizes'
INTENSITY_FIELD_NAME = 'LaserIntensity_MWmm2'  # assuming homogenous intensity for all spots.
HALFTONE_FIELD_NAME = 'intensities'
SPOT_TIMES_FIELD_NAME = 'timing'

class PatternUnit(Unit):
    """Unit from PatternSession"""

    def __init__(self, unit_id, spiketimes, rating, session: 'PatternSession'):
        super(PatternUnit, self).__init__(unit_id, spiketimes, rating, session)
        self.session = session  # for type hinting.

    def plot_spots(self, pre_ms, post_ms, binsize_ms, sequences=None, axis=None, label='', color=None,
                   alpha=.8, linewidth=2, linestyle='-', convolve='gaussian') -> plt.Axes:
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
        :return: axis
        """
        offset_x = (pre_ms + post_ms) * 1.05
        offset_y = 200.
        xs = []
        ys = []

        if axis is None:
            axis = plt.axes()  # type: Axes

        if not sequences:
            sequences = self.session.sequence_dict.keys()
        for seq in sequences:
            times = self.session.sequence_dict[seq]
            if len(seq) == 1 and len(seq.frames) == 1:
                spot = seq.frames[0].spots[0]
                p_x = spot.x - self.session.extents['x'][0]
                p_y = spot.y - self.session.extents['y'][0]
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
        return axis


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
        self._extents = None
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
        protocol_name = meta_loaders._get_voyeur_protocol_name(meta_file)

        if protocol_name == 'patternstim_2AFC_ephys':
            start_frame = 1  # these recordings have a blank start frame.
        else:
            start_frame = 0

        for i in range(n_trs - 1):
            st, tr = trials[i]
            spot_coord_str = tr[SPOT_FIELD_NAME].decode()  # spots are stored as a list of x,y coordinates (ie [[2,3], [5,3]])
            if spot_coord_str:
                spot_size_str = tr[SPOTSIZE_FIELD_NAME].decode()  # decode transforms bytes to str
                spot_times = tr[SPOT_TIMES_FIELD_NAME].decode()
                spot_halftone_intensity_str = tr[HALFTONE_FIELD_NAME].decode()
                spot_halftone_intensity = meta_loaders.poly_spot_to_list(spot_halftone_intensity_str)
                spot_coordinates = meta_loaders.poly_spot_to_list(spot_coord_str)
                spot_times = meta_loaders.poly_spot_to_list(spot_times)
                spot_sizes = meta_loaders.poly_spot_to_list(spot_size_str)
                spot_size_dict = {}
                spot_halftone_intensity_dict = {}
                for coord, spotsize, spot_i in zip(spot_coordinates, spot_sizes, spot_halftone_intensity):
                    coord = tuple(coord)  # hashable
                    spot_size_dict[coord] = spotsize
                    spot_halftone_intensity_dict[coord] = spot_i
                frame_spots, frame_times = _make_sparse_frame_list(spot_coordinates, spot_times)
                try:
                    laser_intensity = tr[INTENSITY_FIELD_NAME]  # type: float
                except ValueError:  # very first recordings don't have this field, and were recorded with this power.
                    print('Warning, no intensity field found, using default of 187.')
                    laser_intensity = 187.
                nd = allstarts[i + 1]
                pulse_idxes = np.where((pulse_starts >= st) & (pulse_starts < nd))[0]
                numpulses = len(pulse_idxes)
                sequence_frames = list()
                sequence_times = list()
                if numpulses - start_frame == len(frame_spots):
                    # pulses include a start blank frame (sometimes), and end blank frame. These are inferred
                    # and are not listed as spots.
                    assert meta_loaders.find_list_depth(frame_spots) < 4
                    # even if we are presenting multiple spots in a frame (or pulse), they will be
                    # represented in a list. So we'll have a spot lists like: [[[2,3],[4,3]], [[2,1], [3,4]]],
                    # where we have 2 frames each with 2 spots, but the length of laser_spots is 2.
                    for j in range(start_frame, numpulses - 1):   # skip the blank first frame if it's there.
                        # we're going to ignore the first and last stim spots (see above).
                        i_p = pulse_idxes[j]
                        i_frame = j - start_frame  # the spots start at index 0
                        t_frame = pulse_starts[i_p]
                        frame_coordinates = frame_spots[i_frame]
                        frame_coordinates.sort()  # sort so that we guarantee consistent ordering of spot lists with same spots
                        sfs = []
                        for coords in frame_coordinates:
                            coords = tuple(coords)
                            spot_halftone = spot_halftone_intensity_dict[tuple(coords)] / 255.
                            spot_intensity = laser_intensity * spot_halftone
                            sz = spot_size_dict[coords]
                            spt = Spot(coords, sz, spot_intensity)
                            self.unique_spots.add(spt)
                            sfs.append(spt)
                        sfs = tuple(sfs)
                        frame = Frame(sfs)
                        sequence_frames.append(frame)
                        sequence_times.append(t_frame)
                    blank_pulse = pulse_starts[pulse_idxes[j + 1]]
                    sequence_times.append(blank_pulse)
                    sequence = FrameSequence(tuple(sequence_frames), tuple(sequence_times), tuple(frame_times))
                    sequence_start_time = sequence_times[0]
                    if sequence in sequence_dict.keys():
                        sequence_dict[sequence].append(sequence_start_time)
                    else:
                        sequence_dict[sequence] = [sequence_start_time]
                    self.sequences.append(sequence)
                else:
                    print('warning {} frames found in trial {} and {} pulses found. Discarding trial.'.format(
                        len(frame_spots), tr['trialNumber'], numpulses
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

    @property
    def extents(self):
        if self._extents is None:
            xs = [x[0] for x in self.unique_coordinates]
            ys = [x[1] for x in self.unique_coordinates]
            self._extents = {
                'x': (min(xs), max(xs)),
                'y': (min(ys), max(ys))
            }
        return self._extents

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
    def __init__(self, frames: Tuple['Frame'], frametimes: Tuple[int], frametimes_relative: Tuple[int]):
        """
        Sequence of frames.

        :param frames: tuple of Frame models.
        :param frametimes: recording time where each frame is presented.
        :param frametimes_relative: This should be the timing of the frames relative to
        inhalation onset. This is used to determine whether the sequence is unique or an instance of an
        existing frame.
        """

        if not len(frames) == len(frametimes) - 1:  # 1 extra frametime encoding the off time.
            raise ValueError('Number of frames must be consistent with number of frametimes')
        if not type(frames) == tuple == type(frametimes):
            raise ValueError('Must use tuples (hashablility)')
        if frametimes_relative is not None and len(frametimes) != len(frametimes_relative):
            raise ValueError('Number of frames and relative frametimes not consistent.')
        assert all([type(x) == Frame for x in frames])
        # assert all([type(x) == int for x in frametimes])
        self.frames = frames
        self.frametimes = frametimes  # absolute frametimes relative to recording time.
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
        # we're using the relative frametimes here. obviously absolute frametimes will be different between
        # frame presentations.
        return hash((self.frames, self.frametimes_relative))

    def __eq__(self, other):
        return self.frametimes_relative == other.frametimes_relative and self.frames == other.frames

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
    def __init__(self, spots: Tuple['Spot']):
        """
        Frames are made up of Spots. A frame can contain as many spots as needed.
        
        :param spots: tuple containing Spot objects.
        """
        assert type(spots) == tuple
        self.spots = spots
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

    def __init__(self, coordinates: Tuple[int], size: float, intensity: float):
        """
        :param coordinates: tuple of (y, x) - REVERSE CARTESIAN.
        :param size: size of spot.
        :param intensity: laser intensity in mW mm-2.
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


def _make_sparse_frame_list(spot_list, spot_timing):
    """
    Makes frame sequence representation out of spot + spot time lists. By EC.

    Each frame is a list of spots that are active at a given time. 
    The sequence is made up of a list of frames.

    For example (+ is on, - is off):
    spot1 (x1, y1):  +++++------
    spot2 (x2, y2):  ---+++++---
    
    Would have a sequence: [[[x1,y1]], [[x1,y1], [x2, y2]], [[x2, x2]], []] 
    
    The last frame in the sequence is empty - no spots are on.

    :param spot_list: [[x1,y1], [x2,y2]] 
    :param spot_timing: [t1, t2]
    :return:  frame sequence, frame times 
    """

    nspots = len(spot_list)
    spot_timing = np.array(spot_timing)

    frame_times = np.unique(spot_timing)

    frame_array = np.zeros([nspots, len(frame_times)])
    for i in range(len(frame_times)):
        switch_time = frame_times[i]
        spots_present = (switch_time >= spot_timing[:, 0]) * (switch_time < spot_timing[:, 1])
        frame_array[:, i] = spots_present

    frame_list = []
    for i in range(len(frame_times)):
        spots_present = frame_array[:, i]
        spots_present = np.where(spots_present == True)[0]
        spots_in_frame = [spot_list[x] for x in spots_present]
        frame_list.append(spots_in_frame)
    assert len(frame_list) == len(frame_times)
    return frame_list, frame_times
