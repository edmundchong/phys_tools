import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import phys_tools as et
import seaborn as sns
sns.set_style('ticks')
from sklearn.metrics import auc

KERNEL_SEARCHSPACE = np.logspace(.30, 2.3, 50)
AUC_THRESHOLD = 0.7



def threshold_by_roc_blanks(rec_dict):
    """

    :param rec_dict:
    :return:
    """
    stims = rec_dict['stims']
    units = rec_dict['units']
    # all_inhs = rec_dict['inhalations']
    # all_exhs = rec_dict['exhalations']
    unit_keys = list(rec_dict['units'].keys())
    unit_keys.sort()
    odor_colors = ['y', 'g', 'r']
    # noodor_inhs = et.basic_plotting.get_no_odor_sniffs(all_inhs, all_exhs, stims['fv_ons'], stims['fv_offs'])[0]
    noodor_inhs = et.basic_plotting.get_odor_sniffs(b'', 0., stims)[0]
    inhs_by_odor = dict()
    odors = np.unique(stims['odors'])
    for o in odors:
        inhs_by_conc = {}
        concs = np.unique(stims['odorconcs'][stims['odors'] == o])
        for c in concs:
            inhs_by_conc[c] = et.basic_plotting.get_odor_sniffs(o ,c, stims)[0]
        inhs_by_odor[o] = inhs_by_conc

    latencies_by_unit = {}
    for k, unit in units.items():
        latencies_by_odor = {}
        r_no = et.basic_plotting.get_rasters(noodor_inhs, unit, 25 * 300, decimate_factor=25)
        for o, inhs_by_conc in inhs_by_odor.items():
            latencies_by_conc = {}
            for c, odor_inhs in inhs_by_conc.items():
                r_od = et.basic_plotting.get_rasters(odor_inhs, unit, 25 * 300, decimate_factor=25)
                aucs = compare_rocs(r_od, r_no,)
                if aucs.max() > AUC_THRESHOLD:
                    w = np.where(aucs == aucs.max())[0][0]
                    kw = KERNEL_SEARCHSPACE[w]
                    kernel = sp.stats.expon.pdf(np.linspace(0,6, kw))
                    best_thresh = optimize_ROC(r_od, r_no, kernel)
                    latencies = get_latencies_convolution(r_od, kernel, best_thresh)
                    latencies_by_conc[c] = latencies
                else:
                    latencies_by_conc[c] = False  # we need to keep track of the non-responses too.
            latencies_by_odor[o] = latencies_by_conc
        latencies_by_unit[k] = latencies_by_odor

    return latencies_by_unit



def threshold_by_roc(rec_dict):
    """

    :param rec_dict:
    :return:
    """
    stims = rec_dict['stims']
    units = rec_dict['units']
    all_inhs = rec_dict['inhalations']
    all_exhs = rec_dict['exhalations']
    unit_keys = list(rec_dict['units'].keys())
    unit_keys.sort()
    odor_colors = ['y', 'g', 'r']
    noodor_inhs = et.basic_plotting.get_no_odor_sniffs(all_inhs, all_exhs, stims['fv_ons'], stims['fv_offs'])[0]
    inhs_by_odor = dict()
    odors = np.unique(stims['odors'])
    for o in odors:
        inhs_by_conc = {}
        concs = np.unique(stims['odorconcs'][stims['odors'] == o])
        for c in concs:
            inhs_by_conc[c] = et.basic_plotting.get_odor_sniffs(o ,c, stims)[0]
        inhs_by_odor[o] = inhs_by_conc

    latencies_by_unit = {}
    for k, unit in units.items():
        latencies_by_odor = {}
        r_no = et.basic_plotting.get_rasters(noodor_inhs, unit, 25 * 300, decimate_factor=25)
        for o, inhs_by_conc in inhs_by_odor.items():
            latencies_by_conc = {}
            for c, odor_inhs in inhs_by_conc.items():
                r_od = et.basic_plotting.get_rasters(odor_inhs, unit, 25 * 300, decimate_factor=25)
                aucs = compare_rocs(r_od, r_no,)
                if aucs.max() > AUC_THRESHOLD:
                    w = np.where(aucs == aucs.max())[0][0]
                    kw = KERNEL_SEARCHSPACE[w]
                    kernel = sp.stats.expon.pdf(np.linspace(0,6, kw))
                    best_thresh = optimize_ROC(r_od, r_no, kernel)
                    latencies = get_latencies_convolution(r_od, kernel, best_thresh)
                    latencies_by_conc[c] = latencies
                else:
                    latencies_by_conc[c] = False  # we need to keep track of the non-responses too.
            latencies_by_odor[o] = latencies_by_conc
        latencies_by_unit[k] = latencies_by_odor

    return latencies_by_unit



def rasters_to_rocAuc(odor_rasters, no_odor_rasters, kernel, plot=False):
    """

    :param odor_rasters:
    :param no_odor_rasters:
    :param kernel:
    :param plot:
    :return:
    """
    odor_maxes = np.zeros(len(odor_rasters))
    no_odor_maxes = np.zeros(len(no_odor_rasters))
    for i in range(len(odor_rasters)):
        r_conv = np.convolve(odor_rasters[i], kernel, 'valid')
        odor_maxes[i] = r_conv.max()
    for i in range(len(no_odor_rasters)):
        r_conv = np.convolve(no_odor_rasters[i], kernel, 'valid')
        no_odor_maxes[i] = r_conv.max()
    threshold_max = odor_maxes.max()
    threshold_min = min([odor_maxes.min(), no_odor_maxes.min()])
    thresholds = np.linspace(threshold_min, threshold_max, 100)
    hits = np.zeros_like(thresholds)
    fas = np.zeros_like(thresholds)
    for i in range(len(thresholds)):
        l = thresholds[i]
        hits[i] = np.sum(odor_maxes > l) / len(odor_maxes)
        fas[i] = np.sum(no_odor_maxes > l) / len(no_odor_maxes)
    a = auc(fas, hits)
    if plot:
        plt.plot(fas, hits, '-', label='AUC={:0.3f}'.format(a))

    return a


def compare_rocs(odor_rasters, no_odor_rasters, plot=False):
    kernel_bw_searchspace = KERNEL_SEARCHSPACE
    aucs = np.zeros_like(kernel_bw_searchspace)
    for i, bw in enumerate(kernel_bw_searchspace):
        l = int(bw)
        k = sp.stats.expon.pdf(np.linspace(0, 6, l))
        aucs[i] = rasters_to_rocAuc(odor_rasters, no_odor_rasters, k, plot)
    if plot:
        # plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], '--k')
        plt.show()
    #TODO: try to actually use an optimizer here instead of fixed grid. This objective function is probably not convex though, so this might not work.

    return aucs


def optimize_ROC(odor_rasters, no_odor_rasters, kernel, plot=False):
    """

    :param odor_rasters:
    :param no_odor_rasters:
    :param kernel:
    :param plot:
    :return:
    """
    l_r = len(odor_rasters.T)
    l_k = len(kernel)
    l = max((l_r, l_k)) - min((l_r, l_k)) + 1
    odor_convs = np.zeros((len(odor_rasters), l))
    no_odor_convs = np.zeros((len(no_odor_rasters), l))

    for i in range(len(odor_rasters)):
        odor_convs[i] = np.convolve(odor_rasters[i], kernel, 'valid')
    for i in range(len(no_odor_rasters)):
        no_odor_convs[i] = np.convolve(no_odor_rasters[i], kernel, 'valid')

    threshold_max = max([odor_convs.max(axis=1).max(), no_odor_convs.max(axis=1).max()])
    threshold_min = min([odor_convs.min(axis=1).min(), no_odor_convs.min(axis=1).min()])
    thresholds = np.linspace(threshold_min, threshold_max, 100)
    hits = np.zeros_like(thresholds)
    fas = np.zeros_like(thresholds)
    for i in range(len(thresholds)):
        l = thresholds[i]
        hits[i] = np.sum(odor_convs.max(axis=1) >= l) / len(odor_convs)
        fas[i] = np.sum(no_odor_convs.max(axis=1) >= l) / len(no_odor_convs)

    orig = np.array([0., 1.])
    distances = np.zeros_like(fas)
    for i in range(len(fas)):
        f = fas[i]
        h = hits[i]
        e = sp.spatial.distance.euclidean(orig, np.array([f, h]))
        distances[i] = e
    best_i = np.where(distances == distances.min())[0][0]
    best_thresh = thresholds[best_i]
    if plot:
        plt.plot(fas, hits, '-o')
        plt.plot(fas[best_i], hits[best_i], '*r')
    #TODO: actually use an optimizer here instead of brute force. If we have a good start point, I think this might be convex (but if we start on a flat spot it might not work)

    return best_thresh


def get_latencies_convolution(odor_rasters, kernel, threshold, plot=False):
    """

    :param odor_rasters:
    :param kernel:
    :param threshold:
    :param plot:
    :return:
    """
    l_r = len(odor_rasters.T)
    l_k = len(kernel)
    l = max((l_r, l_k)) - min((l_r, l_k)) + 1
    latencies = np.zeros(len(odor_rasters), dtype=int)

    for i in range(len(odor_rasters)):
        r_conv = np.convolve(odor_rasters[i], kernel, )
        thresholded = r_conv > threshold
        if np.any(thresholded):
            latencies[i] = np.where(thresholded)[0][0]
            if plot:
                plt.plot(r_conv, '-k', alpha=.3)
        else:
            latencies[i] = -1
            if plot:
                plt.plot(r_conv, '-r', alpha=.3)

    if plot:
        plt.plot(plt.xlim(), [threshold] * 2, '--r')
        for lat in latencies[latencies > 0]:
            plt.plot(lat, 0, '.r')
            #     print (latencies)
    return latencies