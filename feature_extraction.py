import numpy as np
import scipy.stats as stats
from scipy import signal
import warnings
from mne.time_frequency import psd_multitaper
from scipy.integrate import simps
from hurst import compute_Hc
from pyentrp import entropy as ent


def extract(epoch, fs, time_before_stimul, amplitude=False, amplitude_P300=False, kurtosis=False,
                         skewness=False, std=False, sampen=False, rms=False, hurst=False, gradient=False,
                         alfa=False, beta=False, theta=False, delta=False, broad_band=False, **kwargs):
    """
    Extract the features of ONE channel given an epoch structure

    Parameters
    ----------
    epoch : narray (float)
        Data for the features calculation
    time_before_stimul: float
        Time before stimulus ( 0 sec )
    amplitude: bool
        If True calculate the amplitude
    amplitude_P300: bool
        If True calculate the amplitude
    kurtosis: bool
        If True calculate the kurtosis
    skewness: bool
        If True calculate the skewness
    std: bool
        If True calculate the standard deviation
    sampen: bool
        If True calculate the Sample Entropy
    rms: bool
        If True calculate the RMS
    hurst: bool
        If True calculate the Hurst exponent
    gradient: bool
        If True calculate the gradient
    alfa: bool
        If True calculate the alfa band power
    beta: bool
        If True calculate the beta band power
    theta: bool
        If True calculate the theta band power
    delta: bool
        If True calculate the delta band power
    broad_band: bool
        If True calculate the broad band power

    Returns
    -------
    feat: dict
        Extracted features.

    """

    num_samples, num_epochs = epoch.shape

    # Features initialization
    feat = {}
    feat_amplitude = []
    feat_amplitude_P300 = []
    feat_kurt = []
    feat_skew = []
    feat_std = []
    feat_rms = []
    feat_gradient = []
    feat_hurst = []
    feat_sampen = []
    feat_alfa = []
    feat_beta = []
    feat_theta = []
    feat_delta = []
    feat_broad_band = []

    # Retrieving parameters
    win = kwargs.get('window_sec', None)
    method = kwargs.get('method', 'welch')
    relative = kwargs.get('relative', True)
    amplitude_norm = kwargs.get('amplitude_norm', 'median')
    order = kwargs.get('order', 2)

    # Calculate features for each channel
    for ep in range(num_epochs):

        # Pick the current channel and good epochs
        current_epoch = epoch[:, ep]

        # UNIVARIATE FEATURES
        if amplitude:

            # Calculating the normalization factor
            if amplitude_norm == 'median':
                norm_factor = np.median(current_epoch)
            elif amplitude_norm == 'mean':
                norm_factor = np.mean(current_epoch)
            elif amplitude_norm is None:
                norm_factor = 1
            else:
                raise Exception("Unknown normalization factor")

            max_amp = np.max(current_epoch)
            min_amp = np.min(current_epoch)
            feat_amplitude.append((max_amp - min_amp) / norm_factor)

        # Take the maximum value found in the range 300-500 - baseline
        if amplitude_P300:

            # Calculating the normalization factor
            if amplitude_norm == 'median':
                norm_factor = np.median(current_epoch)
            elif amplitude_norm == 'mean':
                norm_factor = np.mean(current_epoch)
            elif amplitude_norm is None:
                norm_factor = 1
            else:
                raise Exception("Unknown normalization factor")


            mean_300_500 = np.mean(current_epoch[round(0.300*fs):])
            baseline = np.mean(current_epoch[:round(fs*time_before_stimul)])
            feat_amplitude_P300.append((mean_300_500 - baseline) / norm_factor)

        if kurtosis:
            feat_kurt.append(stats.kurtosis(current_epoch))

        if skewness:
            feat_skew.append(stats.skew(current_epoch))

        if std:
            feat_std.append(np.std(current_epoch))

        if gradient:
            feat_gradient.append(np.mean(np.gradient(current_epoch)))

        if rms:
            feat_rms.append(np.sqrt(np.mean(np.power(current_epoch, 2))))

        if sampen:
            feat_sampen.append(ent.sample_entropy(current_epoch, order, 0.2*np.std(current_epoch))[0])

        if hurst:
            feat_hurst.append(compute_Hc(current_epoch)[0])

        # FREQUENCY FEATURES
        if alfa:
            feat_alfa.append(band_power(current_epoch, fs, 'alfa', window_sec=win, method=method, relative=relative))

        if beta:
            feat_beta.append(band_power(current_epoch, fs, 'beta', window_sec=win, method=method, relative=relative))

        if theta:
            feat_theta.append(band_power(current_epoch, fs, 'theta', window_sec=win, method=method, relative=relative))

        if delta:
            feat_delta.append(band_power(current_epoch, fs, 'delta', window_sec=win, method=method, relative=relative))

        if broad_band:
            feat_broad_band.append(frequency_baseline(current_epoch, time_before_stimul, fs, 'broad_band'))


    # SAVING THE RESULTS
    if amplitude:
        feat['amplitude'] = feat_amplitude.copy()
    if amplitude_P300:
        feat['amplitude_P300'] = feat_amplitude_P300.copy()
    if kurtosis:
        feat['kurtosis'] = feat_kurt.copy()
    if skewness:
        feat['skewness'] = feat_skew.copy()
    if std:
        feat['std'] = feat_std.copy()
    if gradient:
        feat['gradient'] = feat_gradient.copy()
    if rms:
        feat['rms'] = feat_rms.copy()
    if sampen:
        feat['sampen'] = feat_sampen.copy()
    if hurst:
        feat['hurst'] = feat_hurst.copy()
    if alfa:
        feat['alfa'] = feat_alfa.copy()
    if beta:
        feat['beta'] = feat_beta.copy()
    if theta:
        feat['theta'] = feat_theta.copy()
    if delta:
        feat['delta'] = feat_delta.copy()
    if broad_band:
        feat['broad_band'] = feat_broad_band.copy()

    return feat


def frequency_baseline(current_epoch, time_before_stimul, fs, f_range):
    """
    This function will calculate the mean-power content across epochs for a single channel given a certain frequency
    range of interest.

    Parameters
    ----------
    current_epoch: narray (float)
        Data for the features calculation
    time_before_stimul: float
        Time before stimulus ( 0 sec )
    fs: int
        Sampling frequency
    f_range: str
        Range of frequencies for the estimation of the PSD. It can be a EEG band string. 'theta',
        'broad_band'.
        If a int (or float) the PSD will be estimated within the costume range
        (i.e range between 0 and 10 Hz, f_range = [0, 10])

    Returns
    -------
    power: float
        Relative spectral power of the signal
    """
    if isinstance(f_range, str):

        if f_range == 'theta':
            f_low, f_high = 4, 10
            data_channel_epoch = current_epoch[:round(time_before_stimul * fs)].copy()
        elif f_range =='broad_band':
            f_low, f_high = 0.1, 100
            data_channel_epoch = current_epoch[round(0.300*fs):round(450 * fs)].copy()
        else:
            raise Exception("unknown frequency range")

    # Costume range
    elif all((isinstance(x, float) or isinstance(x, int)) for x in f_range) and len(f_range) < 3:
        f_low, f_high = f_range

    else:
        raise Exception("Unknown range format")

    # Here we calculate the frequency content
    win = (2 / f_low) * fs

    freqs, psd = signal.welch(data_channel_epoch, fs, nperseg=win)

    # Mean across epochs
    psd = psd.squeeze()

    # Find intersecting values in frequency vector
    idx = np.logical_and(freqs >= f_low, freqs < f_high)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Computing the power using the Simpson method

    # Relative power
    range_power = simps(psd[idx], dx=freq_res)
    total_power = simps(psd, dx=freq_res)
    power = range_power / total_power


    return power


def band_power(data_channel, fs, f_range, window_sec=None, method='welch', relative=False):
    """

    Parameters
    ----------
    data_channel: Epochs
        Samples of the good epochs of the channel
    fs: int
        Sampling frequency
    f_range: str or int | float
        Range of frequencies for the estimation of the PSD. It can be a EEG band string. 'alfa','beta','delta,'theta',
        'high_gamma'.
        If a int (or float) the PSD will be estimated within the costume range
        (i.e range between 0 and 10 Hz, f_range = [0, 10])
    window_sec: int or None
        Window in seconds used for the estimation of power spectrum with welch
    method: str
        Method used to calculate the PSD. 'welch', 'multitaper'
    relative: bool
        If True calculate the relative band-power content, if False calculate the absolute band-power content

    Returns
    -------
    power: float
        Spectral power of the signal

    """

    if isinstance(f_range, str):
        # Window length for PSD estimation by Welch
        if f_range == 'delta':    # [0.5 - 4]
            low, high = 0.5, 4    # lower and upper limits
        elif f_range == 'theta':  # [4 - 8]
            low, high = 4, 8  # lower and upper limits
        elif f_range == 'alfa':   # [8 - 13]
            low, high = 8, 13  # lower and upper limits
        elif f_range == 'beta':   # [13 - 30]
            low, high = 13, 30  # lower and upper limits
        elif f_range == 'high_gamma':
            low, high = 150, 250  # lower and upper limits

        else:
            raise Exception("unknown frequency range")

    # Costume range
    elif all((isinstance(x, float) or isinstance(x, int)) for x in f_range) and len(f_range) < 3:
        low, high = f_range

    else:
        raise Exception("Unknown range")


    if method == 'welch':
        if window_sec is not None:
            win = window_sec * fs
        else:
            win = (2 / low) * fs

        freqs, psd = signal.welch(data_channel, fs, nperseg=win)

    elif method == 'multitaper':
        psd, freqs = psd_multitaper(data_channel, picks=data_channel.ch_names, verbose='ERROR')
        psd = psd.squeeze().mean(0)
    else:
        raise Exception('unknown PSD estimation method')

    # Find intersecting values in frequency vector
    idx = np.logical_and(freqs >= low, freqs < high)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Computing the power using the Simpson method
    if relative:
        # Relative power
        range_power = simps(psd[idx], dx=freq_res)
        total_power = simps(psd, dx=freq_res)
        power = range_power / total_power
    else:
        # Absolute power
        power = simps(psd[idx], dx=freq_res)

    return power

