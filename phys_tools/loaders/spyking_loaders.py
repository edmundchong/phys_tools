import tables as tb
import numpy as np
from scipy import sparse
import os
from glob import glob


def make_file_paths(dat_path, suffix='1'):
    """
    Returns tuple of default file paths for spyking circus files.
    :param dat_path:
    :param suffix:
    :return:
    """
    basedir, dat = os.path.split(dat_path)
    name = os.path.splitext(dat)[0]
    sortdir = os.path.join(basedir, name)
    templates = os.path.join(sortdir, '{}.templates{}.hdf5'.format(name, suffix))
    result = os.path.join(sortdir, '{}.result{}.hdf5'.format(name, suffix))
    meta = os.path.join(basedir, '{}_meta.h5'.format(name))
    for f in (templates, result, meta):
        if not os.path.exists(f):
            raise FileNotFoundError(f)
    return templates, result, meta


def find_probe_file(dat_path):
    """ finds probe file in session folder. """
    prb_file = None
    basedir, dat = os.path.split(dat_path)
    prb_files = glob(basedir+'/*.prb')
    if len(prb_files) == 1:
        prb_file = prb_files[0]
    elif len(prb_files) > 1:
        print('Multiple prb files found in folder: {}, no probe information is available'.format(basedir))
    elif not prb_files:
        print('No prb file found in {}'.format(basedir))
    return prb_file


def load_spiketimes(template_fn, results_fn, tag_threshold=1., return_tags=False) -> dict:
    """
    loads units from spyking circus template and results files.

    Thresholds:
        O = 0.
        E = 1.
        D = 2.
        C = 3.
        B = 4.
        A = 5.

    :param template_fn: path to template file (contains tags)
    :param results_fn: path to results file (contains spike times)
    :param tag_threshold: loaders templates that are >= this threshold.
    :return: dictionary of unit spiketimes {unit_numbers: spiketimes}
    """

    units = dict()
    tags = load_template_ratings(template_fn)
    unit_numbers = np.where(tags >= tag_threshold)[0]
    with tb.open_file(results_fn) as results:
        spiketimes = results.root.spiketimes
        for num in unit_numbers:
            spikes = results.get_node(spiketimes, 'temp_{}'.format(num)).read()
            if spikes.ndim > 1:
                spikes = spikes.ravel()
            if not spikes.dtype == np.uint64:
                spikes = spikes.astype(np.uint64)
            units[num] = spikes
    return units


def load_spiketimes_unstructured(template_fn, results_fn, tag_threshold=1.):
    unit_spiketimes = []
    tags = load_template_ratings(template_fn)
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


def load_template_ratings(template_fn) -> np.array:
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


def load_templates(template_fn, template_number=None):
    """
    Loads template array *.templates.h5 file. This is the "waveform" used by the template matching algorithm, which is
    stored as a sparse array.

    The templates returned are in a dictionary. The arrays contained in the dictionary are numpy arrays of shape
    (N_electrodes, template_length_samples).

    :param template_fn: path to template file
    :param template_number: if specified, return single template specified.
    :return: dictionary of template arrays or a single template 2-d array (N_electrodes, template_length_samples)

    """

    with tb.open_file(template_fn) as f:
        shp = f.root.temp_shape.read()
        data = f.root.temp_data.read().ravel()
        x = f.root.temp_x.read().ravel()
        y = f.root.temp_y.read().ravel()

    N_elects, N_samps, N_temps = [int(x) for x in shp]  # unpack the shape array.
    temp_sparse = sparse.csc_matrix((data, (x, y)), shape=(N_elects * N_samps, N_temps))  # make the sparse structure
    if template_number is None:
        templates = dict()
        for i in range(N_temps):  # unpack the sparse structure into
            t = temp_sparse[:, i].toarray()
            t.shape = (N_elects, N_samps)
            templates[i] = t
        return templates
    else:
        t = temp_sparse[:, template_number].toarray()
        t.shape = (N_elects, N_samps)
        return t


def load_probe_positions(probe_fn) -> np.array:
    """
    Makes array of x, y positions by active channels. array shape is (N_ch, 2). Each row is an pair of x and y
    coordinates.

    Dead channels (those not appearing in the probes channels list) are omitted from the array!!! As such, this array is
    indexed to be equivalent to the template array loaded by the load_templates method!

    :param probe_fn: path to .prb file.
    :return: array of positions by channel.
    """

    probe = _read_probe(probe_fn)
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


def _read_probe(prb_fn) -> dict:
    """
    Returns dictionary of probe definition.

    :param prb_fn: path to .prb file
    :return:
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