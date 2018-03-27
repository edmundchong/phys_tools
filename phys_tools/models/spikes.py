"""
Module containing classes that interface with spike sorting result structures. These are ostensibly abstract
classes that allow for operation of other data models with a multitude of different spike sorting systems.

Result classes have public attributes:

    unit_ids: numeric identifier for each unit in the result
    spiketimes: list containing a numpy array of spiketimes for every unit in the result
    ratings: list of numeric quality ratings for each unit in the result
    probe: an object defining probe geometry.
    get_waveforms: a method for retrieving the waveforms for units

There is one more data type that allows for retrieval and parsing of information about probe geometry. For
now, an interface is provided for .prb files that are "industry-standard" for use with klustakwik, phy,
spyking-circus, and kilosort.

Probe classes should have the following public attributes:
    positions: a numpy array of shape (Nch, 2), where each row specifies coordinates for a channel (x, y)
"""


import tables as tb
import numpy as np
from scipy import sparse
import os
from glob import glob

class Result:
    pass


class SpykingResult(Result):

    """
    Interface with spyking-circus file output
    """

    def __init__(self, dat_path, suffix='-1', prb_path=None):
        """

        :param dat_path: path to dat file that generated the result.
        :param suffix: suffix appended to the result file (typically "-1" or "-merged"
        :param prb_path: path to the probe file (optional)
        """

        templates_path, results_path = self._gen_filepaths(dat_path, suffix)
        self._templates_path = templates_path
        self._results_path = results_path

        self.unit_ids, self.spiketimes, self.ratings, = self._load_spiketimes(templates_path, results_path, )

        # handle probe loading
        if prb_path is None:
            prb_path = self._find_prb(dat_path)
        self.probe = Prb(prb_path)

    def get_waveforms(self, unit_id=None):
        """
        Loads template array *.templates.h5 file. This is the "waveform" used by the template matching algorithm, which is
        stored as a sparse array.

        The templates returned are in a dictionary. The arrays contained in the dictionary are numpy arrays of shape
        (N_electrodes, template_length_samples).

        :param unit_id: if specified, return single template specified.
        :return: dictionary of template arrays or a single template 2-d array (N_electrodes, template_length_samples)

        """

        with tb.open_file(self._templates_path) as f:
            shp = f.root.temp_shape.read()
            data = f.root.temp_data.read().ravel()
            x = f.root.temp_x.read().ravel()
            y = f.root.temp_y.read().ravel()

        N_elects, N_samps, N_temps = [int(x) for x in shp]  # unpack the shape array.
        temp_sparse = sparse.csc_matrix((data, (x, y)), shape=(N_elects * N_samps, N_temps))  # make the sparse structure
        if unit_id is None:
            templates = dict()
            for i in range(N_temps):  # unpack the sparse structure into
                t = temp_sparse[:, i].toarray()
                t.shape = (N_elects, N_samps)
                templates[i] = t
            return templates
        else:
            t = temp_sparse[:, unit_id].toarray()
            t.shape = (N_elects, N_samps)
            return t

    @staticmethod
    def _gen_filepaths(dat_path, suffix):
        """
        Generates file paths for spyking circus sorting results. Checks for file existance.  Returns tuple of
        file paths for spyking circus files.

        :param dat_path: Path to the dat file that generated the sorting result.
        :param suffix: suffix of template and result file (typically "-1" or "-merged")
        :return tuple of paths to the files (template, result)
        """
        basedir, dat = os.path.split(dat_path)
        name = os.path.splitext(dat)[0]
        sortdir = os.path.join(basedir, name)
        templates = os.path.join(sortdir, '{}.templates{}.hdf5'.format(name, suffix))
        result = os.path.join(sortdir, '{}.result{}.hdf5'.format(name, suffix))
        for f in (templates, result):
            if not os.path.exists(f):
                raise FileNotFoundError(f)
        return templates, result

    def _find_prb(self, dat_path):
        """ finds probe file in session folder. """
        prb_file = None
        basedir, dat = os.path.split(dat_path)

        prb_files = glob(os.path.join(basedir, '*.prb'))

        if len(prb_files) == 1:
            prb_file = prb_files[0]
        elif len(prb_files) > 1:
            print('Multiple prb files found in folder: {}, no probe information is available'.format(basedir))
        elif not prb_files:
            print('No prb file found in {}'.format(basedir))
        return prb_file

    def _load_spiketimes(self, template_fn, results_fn, tag_threshold=1.):
        """

        :param template_fn:
        :param results_fn: path to filename
        :param tag_threshold: units rated less than this threshold will not be loaded. (0 is O, or unclassed)
        :return:
        """
        unit_spiketimes = []
        tags = self._load_ratings(template_fn)
        unit_numbers = np.where(tags >= tag_threshold)[0]
        unit_ratings = tags[tags >= tag_threshold]
        with tb.open_file(results_fn) as results:
            spiketimes = results.root.spiketimes
            for num in unit_numbers:
                spikes = results.get_node(spiketimes, 'temp_{}'.format(num)).read()
                if spikes.ndim > 1:
                    spikes = spikes.ravel()
                if not spikes.dtype == np.uint64:
                    spikes = spikes.astype(np.uint64)
                unit_spiketimes.append(spikes)
        assert len(unit_numbers) == len(unit_spiketimes) == len(unit_ratings)
        return unit_numbers, unit_spiketimes, unit_ratings

    def _load_ratings(self, template_fn) -> np.array:
        """
        Returns array of tags made by spyking circus matlab GUI.
        0 - unclassified
        1 - "E"
        2 - "D" MU
        3 - "C" Bad SU - probably missing some spikes or contaminated
        4 - "B" Good SU
        5 - "A" Great SU - might have every spike

        :param template_fn: Filename to template file
        :return:
        """
        with tb.open_file(template_fn) as templates:
            tags = templates.root.tagged.read()
            if tags.ndim > 1:
                tags = tags.ravel()
        return tags


class Prb:
    """
    Obj for loading electrode site position information for .prb files used by Phy and Spyking-Circus packages.
    """

    def __init__(self, prb_path):
        self._probe = self._read_prb(prb_path)
        self.positions = self._parse_prb_positions(self._probe)

    @staticmethod
    def _read_prb(prb_fn) -> dict:
        """
        Returns dictionary of probe definition.

        :param prb_fn: path to .prb file
        :return: dictionary
        """

        pr = {}  # temporary dictionary that will be used to execute the probe file.
        probe = {}  # dictionary that will contain only relevant probe file values.
        if not os.path.exists(prb_fn):
            print("The probe file can not be found")
        try:
            with open(prb_fn) as f:
                probetext = f.read()
            exec(probetext, pr)
        except Exception as ex:
            print("Something wrong with the syntax of the probe file:\n" + str(ex))
        key_flags = ['total_nb_channels', 'radius', 'channel_groups']

        for key in key_flags:  # just need to get rid of the extraneous global shit.
            probe[key] = pr[key]

        return probe

    @staticmethod
    def _parse_prb_positions(probe) -> np.array:
        """
        Makes array of x, y positions by active channels. array shape is (N_ch, 2). Each row is an pair of x and y
        coordinates.

        Dead channels (those not appearing in the probes channels list) are omitted from the array!!! As such, this array is
        indexed to be equivalent to the template array loaded by the get_waveform method!

        :param probe: dictionary of probe values #todo: enumerate spec.
        :return: array of positions by channel. (the ith row is the position of the ith channel)
        """

        channel_groups = probe['channel_groups']

        # Make a set of all the active channels and a dictionary of the channels and their positions.
        pos_dict = {}
        all_chs = set()
        for grp in channel_groups.values():
            chs = grp['channels']  # we can't just use all of the positions because of dead channels are censored in template array.
            geometry = grp['geometry']
            for ch in chs:
                if ch in all_chs:
                    raise ValueError('channel number {} occurs multiple times in this probe!'.format(ch))
                else:
                    all_chs.add(ch)
                    pos_dict[ch] = geometry[ch]
        channels = list(all_chs)
        channels.sort()
        N_chs = len(channels)

        # use the position dictionary to make an array of x and y positions.
        pos_array = np.zeros((N_chs, 2))
        for i, ch in enumerate(channels):
            pos = pos_dict[ch]
            pos_array[i, :] = pos
        return pos_array


SPIKE_TYPES = {
    'spyking-circus': SpykingResult,
}