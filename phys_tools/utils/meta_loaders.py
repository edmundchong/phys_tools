import tables as tb
import numpy as np
import datetime

# these are strings that are used to define fieldnames.
# If these change for different filetypes, you can change them here easily.
ODOR_FIELD = "olfas:olfa_{}:odor"
ODOR_CONC_FIELD = "odorconc"
TRIAL_NUM_FIELD = "trialNumber"
INHALATION_NODE = "/Events/inhalations_{}"
EXHALATION_NODE = "/Events/exhalations_{}"
SNIFF_NODE = "/Streams/sniff"


def _findfirst(start, array: np.array):
    a = array >= start
    i = 0
    n = len(array)
    while i < n - 1 and not a[i]:
        i += 1
    if i == n - 1:
        return None
    else:
        return a[i]


def load_sniff_events(meta_file: tb.File):
    """
    get sniffs form meta file.

    if multiple sniff event nodes are present (ie inhalations_1

    :param meta_file: tb.File object
    :return: (inhales, exhales)
    """
    assert type(meta_file) == tb.File
    f = meta_file
    try:
        inh_nodenames = list()
        for i in range(5):
            nd_str = INHALATION_NODE.format(i)
            if f.__contains__(nd_str):
                inh_nodenames.append(nd_str)
        if not inh_nodenames:
            raise (tb.NoSuchNodeError())
        else:
            inhales = f.get_node(inh_nodenames[-1]).read()
    except tb.NoSuchNodeError:
        print('no inhalation nodes found in {}'.format(INHALATION_NODE, f.filename))
        inhales = None
    try:
        exh_nodenames = list()
        for i in range(5):
            nd_str = EXHALATION_NODE.format(i)
            if f.__contains__(nd_str):
                exh_nodenames.append(nd_str)
        if not exh_nodenames:
            raise (tb.NoSuchNodeError())
        else:
            exhales = f.get_node(exh_nodenames[-1]).read()
    except tb.NoSuchNodeError:
        print('no exhalation nodes found in {}'.format(EXHALATION_NODE, f.filename))
        exhales = None

    return inhales, exhales


def _load_voyeur_trials_by_run(meta_file: tb.File) -> list:
    """
    loads all voyeur trial tables and returns them in a sorted list.
    :param meta_file: meta tb.File object
    :return: list of all trials
    """
    f = meta_file
    behavior_node_names = list(f.root.Voyeur._v_children.keys())
    behavior_node_dtg = [_get_date_from_voyeur_name(x) for x in behavior_node_names]
    _, behavior_node_names = zip(*sorted(zip(behavior_node_dtg, behavior_node_names)))
    all_trials = []

    for name in behavior_node_names:
        n = f.get_node('/Voyeur/{}'.format(name))
        run_trials = n.Trials.read()
        all_trials.append(run_trials)
    return all_trials


def _get_date_from_voyeur_name(nodename):
    """
    Parses node names with the following format:
        "***MOUSEDATA***Dyyyy_mm_ddThh_mm_ss"
    
    :param nodename: 
    :return: 
    """

    _, dt = nodename.split('D')
    d_str, t_str = dt.split('T')
    y, m, d = [int(x) for x in d_str.split('_')]
    h, mn, s = [int(x) for x in t_str.split('_')[:3]]  # beh would be the 4th element
    return datetime.datetime(y, m, d, h, mn, s)


def _get_voyeur_protocol_name(meta_file: tb.File) -> str:
    f = meta_file
    b_nodes = f.list_nodes('/Voyeur')
    name = ''
    if b_nodes:
        n = b_nodes[0]  # type: tb.Group
        name = n._v_attrs['arduino_protocol_name'].decode()
    return name


def _load_ephys_trialstarts_by_run(meta_file: tb.File) -> list():
    """
    Breaks trial numbers into putative runs. Within runs, the following condition is always met:
    trial[i+1] > trial[i]. If this condition is not met, then we are starting a new run.

    This returns a structure: [[r1_t1, ..., r1_tN], [r2_t1, ..., r2_tN]]

    :param meta_file: meta tb.File object.
    :return: list of runs, each of which contains a list of trial starts (time, trial_number)
    """
    trialstarts = meta_file.root.Events.trial_starts.read()
    ts_by_run = []
    ts_run = []
    curr_trial = -1
    for st in trialstarts:
        t, tnum = st
        if tnum > 0:
            if curr_trial > tnum or curr_trial == tnum == 1:
                # we've had restart? I don't like that this is not explicit. maybe we can assume restarts start at "1"
                ts_by_run.append(ts_run)
                ts_run = []
                curr_trial = -1
            elif curr_trial == st[1]:
                raise ValueError('duplicate trial numbers in sequence.')
            ts_run.append(st)
            curr_trial = tnum
    ts_by_run.append(ts_run)
    return ts_by_run


def load_aligned_trials(meta_file: tb.File) -> list:
    """
    Loads ephys trial starts and behavior trial starts. Aligns the two. Returns a list of trial start times with
    corresponding lines from voyeur behavior files.

    Runs are not enumerated in the returned structure. In other words, runs are handled correctly, but
    trials are returned in the sequence they occurred regardless of run.

    :param meta_file: open meta tb.File object
    :return:  [(t1_start_time, t1_table_row), ... (tN_start_time, tN_table_row)]
    """
    voyeur_tbr = _load_voyeur_trials_by_run(meta_file)
    ts_br = _load_ephys_trialstarts_by_run(meta_file)
    if not len(voyeur_tbr) == len(ts_br):
        erst = "Diffrent number of runs is detected in \
            Voyeur nodes ({}) and recorded trial starts ({})".format(len(voyeur_tbr), len(ts_br))
        raise ValueError(erst)

    all_trials = []
    for v_trials, e_trials in zip(voyeur_tbr, ts_br):
        v_trial_nums = v_trials['trialNumber']
        e_trial_nums = np.array([x[1] for x in e_trials])
        intersecting_trial_numbers = np.intersect1d(v_trial_nums, e_trial_nums)
        a = np.setdiff1d(v_trial_nums, e_trial_nums)
        b = np.setdiff1d(e_trial_nums, v_trial_nums)

        if a:
            "Voyeur trials {} were not found in recording."
        if b:
            "Trial numbers: {} found in recording but not in voyeur file."
        for tn in intersecting_trial_numbers:
            _v_i = np.where(v_trial_nums == tn)[0]
            assert (len(_v_i) == 1)
            v_i = _v_i[0]
            _e_i = np.where(e_trial_nums == tn)[0]
            assert (len(_e_i) == 1)
            e_i = _e_i[0]
            tr = v_trials[v_i]
            start_time, _ = e_trials[e_i]
            all_trials.append((start_time, tr))
    return all_trials

def load_odor_stims(meta_fn) -> dict:
    """
    Makes a dictionary containing odor stimulus parameters and the times they occur.

    Stimulus dictionary has the following keys:
    'odors': Array of odor spec strings.
    'odorconcs': Array of odor concentrations.
    'fv_times': Array of times when the final valve opened. Times are in samples.

    The arrays are aligned so that at fv_times[i], odors[i] was presented at odorconcs[i]

    :param meta_fn: path to meta file.
    :return: stimulus dictionary.
    """
    n_excepts=0
    with tb.open_file(meta_fn) as f:

        trialsstarts = f.root.Events.trial_starts.read()
        fv = f.root.Events.finalvalve.read()
        fv_starts = fv[:, 0]
        fv_ends = fv[:, 1]
        # utils inhales/exhales if they are available in the file
        inhales, exhales = load_sniff_events(f)
        all_trials = _load_voyeur_trials_by_run(f)

    all_trials_iter = iter(all_trials)


    # these will be the lists that are made into the final arrays of values!
    odorss = []
    concentrationss = []
    fv_ons = []
    fv_offs = []

    inhss = []
    exhss = []
    # Ok, so we're processing the recording (which may have may runs and voyeur files) as one long item. So how do we
    # know when to switch to the next voyeur trials run? Well, if the trial number (in trialstarts) is less than the
    # previous trial number, we assume that we're moving to the next run.
    last_n = -1  # this keeps track of the previous trialnumber
    trials = None

    for i in range(len(trialsstarts)):
        samp, n = trialsstarts[i]
        if n > 0:  # this will ignore bad trial numbers that we're redacting.
            if n < last_n or trials is None:
                trials = all_trials_iter.__next__()
                trs = trials[TRIAL_NUM_FIELD]
                odors_by_olfa = []
                for j in range(2):
                    try:
                        ods = trials[ODOR_FIELD.format(j)]
                        odors_by_olfa.append(ods)
                    except:
                        pass
                cs = trials[ODOR_CONC_FIELD]
                # Quickly check that all of the trial numbers in the run are unique:
                _tr_set = set(trs)
                assert len(trs) == len(_tr_set), 'There are degenerate trial numbers in the Voyeur trials record.'
            if n == 0:
                n = 1
            last_n = n
            ii = np.where(trs == n)[0][0]

            if i < len(trialsstarts) - 1:
                trial_end = trialsstarts[i + 1][0]
            else:
                trial_end = samp + 10000000
            try:
                fvon = fv_starts[fv_starts > samp][0]
                fvoff= fv_ends[fv_ends > fvon][0]
                if fvon < trial_end and fvoff < trial_end:
                    fv_ons.append(fvon)
                    fv_offs.append(fvoff)
                    concentrationss.append(cs[ii])
                    oos = set()
                    for ods in odors_by_olfa:
                        oo = ods[ii]
                        if not oo == b'None':
                            oos.add(oo)
                    assert len(oos) == 1, 'warning functionality for multiple odors from two olfas has not been added'
                    #TODO: need to add the ability to merge strings or something clever here.
                    odorss.append(oos.pop())
                else:
                    fvon = None
                    pass
            except IndexError as e:
                fvon = None
                # n_excepts += 1
                # if n_excepts > 1:
                #     raise e
            if fvon:
                if inhales is not None:
                    inhs = inhales[(inhales > fvon) & (inhales < trial_end)]
                    inhss.append(inhs)
                    if exhales is not None and len(inhs):
                        exhs = exhales[(exhales > inhs[0])]

                        # actually we probably want to just grab the inhalation that is just bigger than the last inhale!!
                        exhss.append(exhs)
                    elif not len(inhs):
                        exhs = np.array([])
                        exhss.append(exhs)

    result = dict(
        fv_ons=np.array(fv_ons),
        fv_offs=np.array(fv_offs),
        odorconcs=np.array(concentrationss),
        odors=np.array(odorss),
        inhales=np.array(inhss),
        exhales=np.array(exhss)
    )

    return result


def load_sniff_trace(meta_f: tb.File) -> np.array:
    """
    loads the sniff trace from meta file
    :param meta_f: open meta file object (tb.File)
    :return: np array of sniff trace
    """

    sniffnode = meta_f.get_node(SNIFF_NODE)
    return sniffnode.read()


def poly_spot_to_list(string: str, _st=0, _return_i=False) -> list:
    """
    Give this a string that looks like a list or list of lists (ie '[[2,2332],[234,12434253]]') and
    it will return list([[2,2332],[243,12434253]]). Yay recursion

    :param string: input string list or list of lists.
    :param _st: index to start processing string (default 0, used for recursion)
    :param _return_i: used for recursion, default False
    """
    if not string:
        return []
    i = _st
    the_list = []
    c = string[i]
    assert c == '[', 'Input must be a list of lists.'
    number_list = []
    while not c == ']':
        i += 1
        c = string[i]
        if c == '[':
            sub_list, i = poly_spot_to_list(string, i, True)
            the_list.append(sub_list)
        if c == ',' and number_list:
            num = int(''.join(number_list))
            the_list.append(num)
            number_list = []
        elif c.isnumeric() and not c == ',':
            number_list.append(c)
    else:
        if number_list:
            num = int(''.join(number_list))
            the_list.append(num)

    if _return_i:  # you need 'i' to update the parent function, but it is ugly to return this by default.
        return the_list, i
    else:
        return the_list


def find_list_depth(list_in, d=0):
    """
    finds the depth of nested lists. ie [2] has depth 1. [[2]] has depth 2. More recursion.
    """
    d += 1  # account for this layer
    depths = [d]
    for i in list_in:
        if type(i) == list:
            depths.append(find_list_depth(i, d))  # record the depths of the subsequent layers
    return max(depths)  # only return the maximum depth from the layer.

# the below is deprecated.
# def load_sniffs(meta_fn):
#     """
#     Loads inhalations and exhalations from meta file.
#
#     :param meta_fn: path to meta file
#     :return: np arrays (inhalations, exhalations)
#     """
#
#     with tb.open_file(meta_fn) as f:
#         try:
#             inh_nodenames = list()
#             for i in range(5):
#                 nd_str = INHALATION_NODE.format(i)
#                 if f.__contains__(nd_str):
#                     inh_nodenames.append(nd_str)
#             if not inh_nodenames:
#                 raise(tb.NoSuchNodeError())
#             else:
#                 inhales = f.get_node(inh_nodenames[-1]).read()
#         except tb.NoSuchNodeError:
#             inhales = np.array([])
#         try:
#             exh_nodenames = list()
#             for i in range(5):
#                 nd_str = EXHALATION_NODE.format(i)
#                 if f.__contains__(nd_str):
#                     exh_nodenames.append(nd_str)
#             if not exh_nodenames:
#                 raise (tb.NoSuchNodeError())
#             else:
#                 exhales = f.get_node(exh_nodenames[-1]).read()
#         except tb.NoSuchNodeError:
#             exhales = np.array([])
#
#     return inhales, exhales