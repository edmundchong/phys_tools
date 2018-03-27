import logging

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm, trange
import numba as nb

MAX_N = 30000
MAX_TRIGGERS = 30000

def _parse_serial(serial_stream, fs=25000., word_len=2, baudrate=300, threshold=None):
    """

    :param serial_stream: array containing serial stream.
    :param fs: sampling frequency of serial stream
    :param word_len: number of bytes per word (ie 16 bit int is 2 bytes)
    :param baudrate: baudrate of serial transmission.
    :param threshold: threshold to use to extract high vs low states. If None, use stream.max / 2
    :return: list of tuples [(time, decoded_number)]
    """

    # NOTE: The bit spacing here is only approximately right due to a problem with arduino timing that makes the endbit
    # between bytes longer than they should be (20% for 300 baud). Ideally we should parse bytes individually, but this
    # works (at least with this baudrate). You would have to be 50% off before this should cause a problem because it is
    # reading the bits from the center of where they should be.

    if serial_stream.ndim == 2:
        try:
            shortdim = serial_stream.shape.index(1)
            if shortdim == 1:
                serial_stream = serial_stream[:, 0]
            else:
                serial_stream = serial_stream[0, :]
        except ValueError:
            raise ValueError('_parse_serial input must be a 1d array or 2d with dimension of length 1.')

    start_bit = 1
    end_bit = 1
    bit_len = float(fs)/baudrate

    if threshold is None:
        # this is a really crappy way of finding the threshold, to be sure.
        threshold = np.max(serial_stream)/2

    log_stream = (serial_stream > threshold)
    edges = np.convolve(log_stream, [1,-1])
    ups = np.where(edges == 1)[0]
    downs = np.where(edges == -1)[0]

    #check to see if baudrate/fs combo is reasonable by looking for bits of the length specified by them in the stream.
    diff = (downs-ups) / bit_len
    cl = 0
    for i in range(1,4):  # look for bit widths of 1,2,3 consecutive.
        diff -= 1.
        g = diff > -.1
        l = diff < .1
        cl += np.sum(g*l)
    if cl < 4:
        print ('WARNING: bits do not appear at timings consistent with selected \nsampling frequency and baud rate ' \
              'combination: (fs: %i, baud: %i). This warning may be erroneous for short recordings.' % (fs, baudrate))

    start_id = downs-ups > int(bit_len * word_len * (8 + start_bit + end_bit))
    start_samples = downs[start_id]

    # Arduino software serial can start initially low before the first transmission, so the start detector above will fail.
    # This means that the initial start bit is masked, as the signal is already low. We're going to try to find the
    # end bit of this ridiculous first word, and work backward to find where this masked start bit occured.
    if not log_stream[0]:
        try:
            for i in range(len(ups)-1):
                if (ups[i+1] - ups[i]) > (bit_len * word_len * (8 + start_bit + end_bit)):

                    firstword_end = ups[i]
                    break
        except IndexError:
            raise ValueError('Serial parsing error: stream starts low, but cannot find end of first word.')
        bts = word_len * (8+start_bit+end_bit) - 1 # end bit of the last byte IS the up, so we shouldn't count it.
        firstword_start = firstword_end - (bts*bit_len)
        firstword_start = np.int(firstword_start)
        start_samples = np.concatenate(([firstword_start], start_samples))

    word_bits = bit_len * word_len * (8+start_bit+end_bit)
    bit_sample_spacing = np.linspace(bit_len/2, word_bits - bit_len/2, word_len * 10)
    bit_sample_spacing = bit_sample_spacing.round().astype(int)

    _bytes = np.empty((word_len, 8), dtype=bool)

    trial_times = []
    for start in start_samples[:-1]:
        bit_samples = bit_sample_spacing + start
        bits = log_stream[bit_samples]
        for i in range(word_len):
            # This strips serial protocol bits assuming that a serial byte looks like: [startbit, 0, 1, ..., 7, endbit]
            # as per normal serial com protocol. In other words: returned _bytes are 8 bit payloads.
            _bytes[i,:] = bits[i*10+1:i*10+9]
        number = _bytes_to_int(_bytes)
        trial_times.append((start, number))
    return trial_times


def _bytes_to_int(bytes, endianness='little'):
    """

    :param bytes: np boolean array with shape == (number_bytes, 8)
    :param endianness of the encoded bytes
    :return:
    """
    bits = bytes.ravel()
    if endianness == 'big':
        bits = np.flipud(bits)
    num = 0
    for e, bit in enumerate(bits):
        num += bit * 2 ** e
    return num

def parse_serial_stream(stream, fs=25000):
    """
    Finds serial numbers if they exist in a stream.
    Uses a more intelligent method to describe the
    threshold value within the stream.
    (fit with 1 or 2 gaussians. If fit w/ 1 is better, no opens. Else, threshold is between the two means)

    :param stream: stream to find threshold crossings.
    :return: list of tuples [(starttimes, decoded number)]
    """
    logging.info("Parsing serial stream...")

    # first determine if our stream is fit better by 1 or two gaussian distributions:
    params = _gaussian_model_comparison(stream)
    if len(params) > 2:
        mu1, mu2, s1, s2 = params
        # if it is fit by 2 gaussians, next check to see if the gaussians are really distinct:
        if np.abs(mu1 - mu2) > s1 * 6:
            threshold = np.min([mu1, mu2]) + np.abs(mu1-mu2)/2
            threshold = threshold.astype(stream.dtype)
        else:
            threshold = None
    else:  # if the stream is fit better by one gaussian, then we don't have any fv opens.
        threshold = None

    if threshold is not None:  #if we have set a threshold above, find the finalvalve opens.
        logging.debug('using threshold {}'.format(threshold))
        trial_times = _parse_serial(stream, fs, threshold=threshold, )
    else:
        trial_times = []
    logging.info('Complete. {} trial starts found.'.format(len(trial_times)))

    return trial_times


# @jit(cache=True)
def findfvopens(stream, *args, threshold=None):
    """
    This very simplistic. Uses threshold value equal to stream.mean() unless you specify a threshold.

    :param stream: the stream you want to threshold
    :param threshold: threshold value.
    :return:
    """
    fv = np.zeros((MAX_TRIGGERS,2), dtype=np.uint64)
    fv_starts = fv[:,0]
    fv_stops = fv[:,1]
    nstarts = 0
    fvopen = False
    lastopen = np.uint64(0)
    if threshold is None:
        threshold = np.mean(stream)
    nstops = 0

    for i in tqdm(range(len(stream)), unit='samp', desc='FV scan'):
        val = stream[i]
        if not fvopen and val > threshold and (val - lastopen > 500):
            fv_starts[nstarts] = i
            nstarts+=1
            fvopen = True
        elif fvopen and val < threshold:
            fv_stops[nstops] = i
            nstops += 1
            fvopen = False
    if nstops:
        return fv[:nstops,:]
    else:
        return np.array([], dtype=np.uint64)


def findfvopens_MC(stream, *args):
    """
    Finds finalvalve openings if they exist in a stream. Uses a more intelligent method to describe the
    threshold value within the stream.
    (fit with 1 or 2 gaussians. If fit w/ 1 is better, no opens. Else, threshold is between the two means)

    :param stream: stream to find threshold crossings.
    :return:
    """

    # first determine if our stream is fit better by 1 or two gaussian distributions:
    params = _gaussian_model_comparison(stream)
    if len(params) > 2:
        mu1, mu2, s1, s2 = params
        # if it is fit by 2 gaussians, next check to see if the gaussians are really distinct:
        if np.abs(mu1 - mu2) > s1 * 6:
            threshold = np.min([mu1, mu2]) + np.abs(mu1-mu2)/2
            threshold = threshold.astype(stream.dtype)
        else:
            threshold = None
    else:  # if the stream is fit better by one gaussian, then we don't have any fv opens.
        threshold = None

    if threshold is not None:  #if we have set a threshold above, find the finalvalve opens.
        logging.debug('using threshold {}'.format(threshold))
        fv_opens = findfvopens(stream, threshold)
    else:

        fv_opens = np.array([], dtype=np.uint64)
    logging.info('Found {} FV opens.'.format(len(fv_opens)))
    return fv_opens


def _gaussian_model_comparison(stream):
    """
    TTL values are modeled as value plus gaussian noise. Find if the stream data is explained better by
    one or two gaussians. We're determining model fit by fitting using MLE and using BIC to compare model
    fits (while penalizing for number of parameters, althought this isn't a big issue.

    :param stream:
    :return:
    """
    N = len(stream)
    maxN = 30000
    if N > maxN:
        dec = int(np.ceil(N/maxN))  # we're guaranteed to get at least 15000 samples here which should be ok.
        assert dec > 0
    else:
        dec = 1

    X = stream[::dec].astype(np.float64)
    N_X = len(X)
    mu_bound1 = np.min(X)
    mu_bound2 = np.max(X)
    mu_st1 = max((mu_bound1+1., 100.))
    mu_st2 = min((mu_bound2-1., 70000.))
    fit1 = _fit_gaussian(X,  μ_bounds=(mu_bound1,mu_bound2))
    fit2 = _fit_sum_of_2gaussians(X,  μ_bounds=(mu_bound1,mu_bound2))
    for f in (fit1, fit2):
        assert f.success

    ic1 = _bic(fit1.fun, 2., N_X)
    ic2 = _bic(fit2.fun, 4., N_X)

    if ic1 < ic2:  # lower BIC wins
        result = fit1.x
        logging.info('1 gaussian fit is better. {}'. format(result))
    else:
        result = fit2.x
        logging.info('2 gaussian fit is better. {}'.format(result))
    return result


def _fit_gaussian(x,
                  x0=(15000., 5.),
                  μ_bounds=(0,30000.),
                  σ_bounds=(.1,8000.)):
    """

    :param x: array of observations to fit (should be float64)
    :param x0: starts (μ, σ)
    :param μ_bounds: bounds for the mean (loc) parameters
    :param σ_bounds: bounds for std parameter (scale)
    :return:
    """
    assert len(x) <= 30000, 'Too many values will make this over/underflow.'
    bounds = (μ_bounds, σ_bounds)

    def nll(args):
        μ, σ = args
        return -norm.logpdf(x, loc=μ, scale=σ).sum()

    return minimize(nll, x0, bounds=bounds)

def _fit_sum_of_2gaussians(x,
                           x0=(400., 15000., 1., 10.),
                           μ_bounds=(0., 70000.),
                           σ_bounds=(.01, 3000.)):
    """

    :param x: array of observations to fit (should be float64)
    :param x0: starts (μ1, μ2, σ1, σ2)
    :param μ_bounds: bounds for the mean (loc) parameters. Will be used for both gaussians.
    :param σ_bounds: bounds for std parameter (scale). Will be used for both gaussians.
    :return:
    """

    assert len(x) <= 30000, 'Too many values will make this over/underflow.'
    bounds = (μ_bounds, μ_bounds, σ_bounds, σ_bounds)

    def nll(args):  #objective function to optimize.
        μ1, μ2, σ1, σ2 = args
        f1 = norm.logpdf(x, loc=μ1, scale=σ1)
        f2 = norm.logpdf(x, loc=μ2, scale=σ2)
        ss = np.logaddexp(f1, f2)  # only way to add these tiny numbers on a computer. read the fn docs.
        return -ss.sum() + np.log(0.5)  # log identity: log(a/2) = log(a) + log(1/2)

    return minimize(nll, x0, bounds=bounds)

def _bic(nlogL, k, n):
    """
    Bayesian Information Criterion
    nlogL is the _negative_ log likelihood of the model.
    k is number of parameters in model.
    n is the number of observations that were used to fit the model. (ie the length of the array)
    """
    return 2. * nlogL + k * np.log(n)


def laseronsets(stream, *args):
    """
    This very simplistic. Uses threshold value equal to half way between the min and max of the stream
    values unless you specify a threshold.

    :param stream: the stream you want to threshold
    :param threshold: threshold value if you want to force it.
    :return: array shape (n laser ons, 2) with the first column being laser ons and the second column being offs
    """
    lsr = np.zeros((MAX_TRIGGERS, 2), dtype=np.uint32)  # big enough for 1.6 days of recordings...
    # lsr_starts = lsr[:, 0]
    # lsr_stops = lsr[:, 1]
    # nstarts = 0
    # lsr_on = False
    # laston = np.uint64(0)

    r = stream.max() - stream.min()
    threshold = stream.min() + r/2.
    # nstops = 0

    # for i in trange(len(stream), unit='samp', desc='laser scan', unit_scale=True):
    #     val = stream[i]
    #     if not lsr_on and val > threshold and (val - laston > 500):
    #         lsr_starts[nstarts] = i
    #         nstarts += 1
    #         lsr_on = True
    #     elif lsr_on and val < threshold:
    #         lsr_stops[nstops] = i
    #         nstops += 1
    #         lsr_on = False

    n_pulses = _findonsets(stream, lsr, threshold)

    if n_pulses:
        return lsr[:n_pulses, :]
    else:
        return np.array([], dtype=np.uint64)

@nb.jit
def _findonsets(stream, out, threshold):
    n_pulses = 0
    lsr_on = False
    # laston = np.uint64(0)

    for i in range(len(stream)):
        val = stream[i]
        if not lsr_on and val > threshold:
            lsr_on = True
            out[n_pulses, 0] = i
        elif lsr_on and val < threshold:
            out[n_pulses, 1] = i
            n_pulses += 1
            lsr_on = False
    return n_pulses



def dmd_frames(stream, fs=30000):
    global MAX_TRIGGERS
    MAX_TRIGGERS=int(1e7)
    return laseronsets(stream, fs)


processors = {
    'finalvalve': findfvopens_MC,
    'trial_starts': parse_serial_stream,
    'laser': laseronsets,
    'dmd_frames': dmd_frames
}


