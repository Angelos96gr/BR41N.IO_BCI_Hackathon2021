import numpy as np
import scipy.stats as stats
from scipy import signal
import warnings
from mne.time_frequency import psd_multitaper
from scipy.integrate import simps
from hurst import compute_Hc
from pyentrp import entropy as ent


def extract(epoch, fs, time_total_stimul, time_before_stimul, amplitude=False, amplitude_P300=False, kurtosis=False,
                         skewness=False, std=False, sampen=False,
                         rms=False, hurst=False, dev_neigh=False, gradient=False, cor_neigh=False,
                         var_neigh=False, alfa=False, beta=False, theta=False, delta=False,
                        base_theta=False, broad_band=False, base_high_gamma=False, **kwargs):


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
    feat_bad_trials = []
    feat_base_high_gamma = []
    feat_base_theta = []
    feat_broad_band = []

    # Retrieving parameters
    win = kwargs.get('window_sec', None)
    method = kwargs.get('method', 'welch')
    relative = kwargs.get('relative', True)
    amplitude_norm = kwargs.get('amplitude_norm', 'median')
    order = kwargs.get('order', 2)
    distance = kwargs.get('distance', 15)
    neigh_distance = kwargs.get('neigh_distance', 2)

    # Calculate features for each channel
    for ep in range(num_epochs):
        # Pick the current channel and good epochs
        current_epoch = epoch[:, ep]

        # List of the bad epochs for the current channel
        #bad_epochs = [ix for ix, x in enumerate(current_channel.bad_trials.T[ch]) if x == 1]

        # Extracting the features using the good epochs
        #data_channel = current_channel.copy().drop(bad_epochs, verbose='ERROR')
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

        if base_theta:
            feat_base_theta.append(frequency_baseline(current_epoch, time_before_stimul, fs, 'theta'))

        if broad_band:
            feat_broad_band.append(frequency_baseline(current_epoch, time_before_stimul, fs, 'broad_band'))

        # Number of bad trials
        #feat_bad_trials.append(len(bad_epochs) / num_epochs)
    """
    # BIVARIATE FEATURES
    if cor_neigh:
        if isinstance(coord, pd.DataFrame):
            feat_cor = feature_cor_neigh_3d(dataset, coord, distance=distance)
        else:
            feat_cor = feature_cor_neigh(dataset, neigh_distance=neigh_distance)

    if var_neigh:
        if isinstance(coord, pd.DataFrame):
            feat_var = feature_var_neigh_3d(dataset, coord, distance=distance)
        else:
            feat_var = feature_var_neigh(dataset, neigh_distance=neigh_distance)

    if dev_neigh:
        if isinstance(coord, pd.DataFrame):
            feat_dev = feature_dev_neigh_3d(dataset, coord, distance=distance)
        else:
            feat_dev = feature_dev_neigh(dataset, neigh_distance=neigh_distance)"""

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
    if base_high_gamma:
        feat['base_high_gamma'] = feat_base_high_gamma.copy()
    if base_theta:
        feat['base_theta'] = feat_base_theta.copy()
    if broad_band:
        feat['broad_band'] = feat_broad_band.copy()
    """if cor_neigh:
        feat['cor_neigh'] = list(feat_cor)
    if var_neigh:
        feat['var_neigh'] = list(feat_var)
    if dev_neigh:
        feat['dev_neigh'] = list(feat_dev)"""

    #feat['bad_epochs'] = feat_bad_trials

    return feat


def feature_cor_neigh(dataset, neigh_distance=2):

    """
    Function used to calculate the mean Pearson correlation coefficient of each channel against the neighboring channels

    Parameters
    ----------
    dataset: mne.Epochs
        Data
    neigh_distance: int
        Number of neighbouring channels on the same rod.
        Example: if 'neigh_distance' is 1, the feature is calculated by considering as the neighbours the immediate
        previous and next channel in the rod

    Returns
    -------
    feat_current: list
        List of the mean Pearson correlation coefficients

    """
    # Extract the usable information from 'dataset'
    data_channel = dataset.get_data()
    num_channels = data_channel.shape[1]
    channels_name = dataset.ch_names
    bad_trials = dataset.bad_trials

    # I want to remove the number from the channels
    no_int = [''] * num_channels
    bool_index = [''] * num_channels
    for j in range(num_channels):
        no_int[j] = ''.join([i for i in channels_name[j] if not i.isdigit()])

    # Initialization of the features and index-i
    feat_current = []
    i = 0

    while i < num_channels:

        # Current electrodes under investigation
        current_electrode = no_int[i]
        for z in range(num_channels):
            bool_index[z] = current_electrode == no_int[z]

        # Extract the channels of the given electrode
        subgroup_channel = [x for x, y in zip(channels_name, bool_index) if y]
        # Extract the numbers of the previously selected channels
        subgroup_number = list(map(lambda sub: int(''.join(
            [n for n in sub if n.isnumeric()])), subgroup_channel))

        # Count the number of channels for the 'current electrode'
        num_ch = len(subgroup_channel)

        for k, value in enumerate(subgroup_number):
            # Range of observation, in the center there will be the channel's number under observation
            range_obs = np.array(range(-neigh_distance, neigh_distance + 1)) + value
            range_obs = [x for x in range_obs if (x > 0) and (x <= max(subgroup_number))]
            range_subset = np.array([subgroup_number.index(x) for x in range_obs if x in subgroup_number])
            # I want to avoid self counting
            range_subset_neigh = np.delete(range_subset, np.where(range_subset == k))
            # If the i don't find any channel in the neighbour put NaN
            if len(range_subset_neigh) == 0:
                correlation = np.nan
                warnings.warn('There are no neighbours within the given distance value')

            else:
                # Pick the current and neighbouring bad epochs vector
                neigh_bad_epochs = bad_trials[:, range_subset + i].sum(axis=1)
                # Consider just the good epoch (i.e. neigh_bad_epochs == 0)
                neigh_good_epochs = [x for x, v in enumerate(neigh_bad_epochs) if v == 0]

                neigh_data = data_channel[:, range_subset_neigh + i, :][neigh_good_epochs, :, :].transpose(1, 2, 0)
                neigh_data = np.reshape(neigh_data, (neigh_data.shape[0], neigh_data.shape[1] * neigh_data.shape[2]), order='F')
                correlation = np.sum(np.corrcoef(data_channel[neigh_good_epochs, i + k, :].flatten(), neigh_data)[0, 1:])\
                              / len(range_subset_neigh)

            feat_current.append(correlation)

        i += num_ch
    return feat_current


def feature_var_neigh(dataset, neigh_distance=1):

    """
    The function will calculate the variance of each channel normalized by the median variance of the neighboring
    channels.

    Parameters
    ----------
    dataset: mne.Epochs
        Data
    neigh_distance: int
        Number of neighbouring channels on the same rod.
        Example: if 'neigh_distance' is 1, the feature is calculated by considering as the neighbours the immediate
        previous and next channel in the rod

    Returns
    -------
    feat_current: list
        List of the variances

    """
    # Extract the usable information from 'dataset'
    data_channel = dataset.get_data()
    num_channels = data_channel.shape[1]
    channels_name = dataset.ch_names
    bad_trials = dataset.bad_trials

    # I want to remove the number from the channels
    no_int = [''] * num_channels
    bool_index = [''] * num_channels
    for j in range(num_channels):
        no_int[j] = ''.join([i for i in channels_name[j] if not i.isdigit()])

    # Initialization of the features and index-i
    feat_current = []
    i = 0

    while i < num_channels:

        # Current electrodes under investigation
        current_electrode = no_int[i]
        for z in range(num_channels):
            bool_index[z] = current_electrode == no_int[z]

        # Extract the channels of the given electrode
        subgroup_channel = [x for x, y in zip(channels_name, bool_index) if y]
        # Extract the numbers of the previously selected channels
        subgroup_number = list(map(lambda sub: int(''.join(
            [n for n in sub if n.isnumeric()])), subgroup_channel))

        # Count the number of channels for the 'current electrode'
        num_ch = len(subgroup_channel)

        for k, value in enumerate(subgroup_number):
            # Range of observation, in the center there will be the channel's number under observation
            range_obs = np.array(range(-neigh_distance, neigh_distance + 1)) + value
            range_obs = [x for x in range_obs if (x > 0) and (x <= max(subgroup_number))]
            range_subset = np.array([subgroup_number.index(x) for x in range_obs if x in subgroup_number])

            # I want to avoid to self counting
            range_subset_neigh = np.delete(range_subset, np.where(range_subset == k))

            # If the i don't find any channel neighbour (expect the channel itself) put NaN
            if len(range_subset_neigh) == 0:
                variance = np.nan
                warnings.warn('There are no neighbours within the given distance value')

            else:
                # Pick the current and neighbouring bad epochs vector
                neigh_bad_epochs = bad_trials[:, range_subset + i].sum(axis=1)
                # Consider just the good epoch (i.e. neigh_bad_epochs == 0)
                neigh_good_epochs = [x for x, v in enumerate(neigh_bad_epochs) if v == 0]

                neigh_data = data_channel[:, range_subset_neigh + i, :][neigh_good_epochs, :, :]
                variance = np.var(data_channel[neigh_good_epochs, i + k, :]) / np.median(np.var(neigh_data, axis=(0, 2)))

            feat_current.append(variance)
        i += num_ch
    return feat_current


def feature_dev_neigh(dataset, neigh_distance=1):

    """
    Function that calculate the deviation of the mean amplitude for each channel against
    the mean amplitude of the neighboring channels

    Parameters
    ----------
    dataset: mne.Epochs
        Data
    neigh_distance: int
        Number of neighbouring channels on the same rod.
        Example: if 'neigh_distance' is 1, the feature is calculated by considering as the neighbours the immediate
        previous and next channel in the rod

    Returns
    -------
    feat_current: list
        List of the calculated deviations

    """
    # Extract the usable information from 'dataset'
    data_channel = dataset.get_data()
    num_channels = data_channel.shape[1]
    channels_name = dataset.ch_names
    bad_trials = dataset.bad_trials

    # I want to remove the number from the channels
    no_int = [''] * num_channels  # todo: use regex
    bool_index = [''] * num_channels
    for j in range(num_channels):
        no_int[j] = ''.join([i for i in channels_name[j] if not i.isdigit()])

    # Initialization of the features and index-i
    feat_current = []
    i = 0
    while i < num_channels:

        # Current electrodes under investigation
        current_electrode = no_int[i]
        for z in range(num_channels):
            bool_index[z] = current_electrode == no_int[z]

        # Extract the channels of the given electrode
        subgroup_channel = [x for x, y in zip(channels_name, bool_index) if y]
        # Extract the numbers of the previously selected channels
        subgroup_number = list(map(lambda sub: int(''.join(
            [n for n in sub if n.isnumeric()])), subgroup_channel))

        # Count the number of channels for the 'current electrode'
        num_ch = len(subgroup_channel)

        for k, value in enumerate(subgroup_number):
            # Range of observation, in the center there will be the channel's number under observation
            range_obs = np.array(range(-neigh_distance, neigh_distance + 1)) + value
            range_obs = [x for x in range_obs if (x > 0) and (x <= max(subgroup_number))]
            range_subset = np.array([subgroup_number.index(x) for x in range_obs if x in subgroup_number])

            # I want to avoid to self counting
            range_subset_neigh = np.delete(range_subset, np.where(range_subset == k))
            # If the i don't find any channel neighbour (expect the channel itself) put NaN
            if len(range_subset_neigh) == 0:
                deviation = np.nan
                warnings.warn('There are no neighbours within the given distance value')

            else:
                # Pick the current and neighbouring bad epochs vector
                neigh_bad_epochs = bad_trials[:, range_subset + i].sum(axis=1)
                # Consider just the good epoch (i.e. neigh_bad_epochs == 0)
                neigh_good_epochs = [x for x, v in enumerate(neigh_bad_epochs) if v == 0]

                neigh_data = data_channel[:, range_subset_neigh + i, :][neigh_good_epochs, :, :]
                deviation = np.mean(data_channel[neigh_good_epochs, i + k, :]) - np.mean(np.mean(neigh_data))

            feat_current.append(deviation)
        i += num_ch
    return feat_current


def frequency_baseline(current_epoch, time_before_stimul, fs, f_range):
    """
    This function will calculate the mean-power content across epochs for a single channel given a certain frequency
    range of interest.

    Parameters
    ----------
    data_channel_epoch: mne.Epochs
        Data
    f_range: str
        Range of frequencies for the estimation of the PSD. It can be a EEG band string. 'theta',
        'high_gamma', 'broad_band'.
        If a int (or float) the PSD will be estimated within the costume range
        (i.e range between 0 and 10 Hz, f_range = [0, 10])
    t_min: float | None
        Min time of interest.
    t_max: float | None
        Max time of interest.

    Returns
    -------
    power: float
        Spectral power of the signal
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


def feature_var_neigh_3d(dataset, coord, distance=15):
    """
    The function will calculate the variance of each channel normalized by the median variance of the neighboring
    channels.
    This function need that Epoch object has the electrode coordinates of the channels

    Parameters
    ----------
    dataset: mne.Epoch
        Data
    coord: pondas.Dataframe
        Dataframe containing the 3D coordinates of the channels
    distance: int
        Distance in mm used to extract the neighbouring channels

    Returns
    -------
    feat_current: list
        List of the variances

    """
    # Extract the usable information from 'dataset'
    data_channel = dataset.get_data()
    channels_name = dataset.ch_names
    bad_trials = dataset.bad_trials

    feat_current = []

    # Metadata info
    meta = coord.loc[coord['name'] == channels_name]

    # Retrieving the coordinate
    coordinates = meta[['x', 'y', 'z']]
    # Re-organize the indexes in order to match later with 'data_channel'
    coordinates = coordinates.reset_index(drop=True)

    for index, ch in enumerate(channels_name):
        dist_all = np.sqrt(np.sum((coordinates - coordinates.iloc[index]) ** 2, axis=1))
        close_channel = dist_all.loc[dist_all < distance]

        # I'll extract the indexes of the the close channel (i'll exclude the channel with 0.0 distance)
        close_indexes = close_channel.loc[close_channel.index != index].index

        # Pick the current and neighbouring bad epochs vector
        neigh_bad_epochs = bad_trials[:, close_channel.index].sum(axis=1)

        # Consider just the good epoch (i.e. neigh_bad_epochs == 0)
        neigh_good_epochs = [x for x, v in enumerate(neigh_bad_epochs) if v == 0]
        channel_subgroup = data_channel[neigh_good_epochs, :, :][:, close_indexes, :]

        # Check if there are no neighbour with the given distance
        if len(channel_subgroup) == 0:
            normalized_variance = np.nan
            warnings.warn('There are no neighbours within the given distance value')
        else:

            normalized_variance = np.var(data_channel[neigh_good_epochs, index, :]) / np.median(np.var(channel_subgroup, axis=(0,2)))

        feat_current.append(normalized_variance)

    return feat_current


def feature_dev_neigh_3d(dataset, coord, distance=15):
    """
    Function that calculate the deviation of the mean amplitude for each channel against
    the mean amplitude of the neighboring channels
    This function need that dataset object have the electrode coordinates of the channels

    Parameters
    ----------
    dataset: mne.Epoch
        Data
    coord: pondas.Dataframe
        Dataframe containing the 3D coordinates of the channels
    distance: int
        Distance in mm used to extract the neighbouring channels

    Returns
    -------
    feat_current: list
        List of the calculated deviations

    """

    # Extract the usable information from 'dataset'
    data_channel = dataset.get_data()
    channels_name = dataset.ch_names
    bad_trials = dataset.bad_trials

    # Initialization
    feat_current = []

    meta = coord.loc[coord['name'] == channels_name]

    # Retrieving the coordinate
    coordinates = meta[['x', 'y', 'z']]
    # Re-organize the indexes in order to match later with 'data_channel'
    coordinates = coordinates.reset_index(drop=True)

    for index, ch in enumerate(channels_name):
        dist_all = np.sqrt(np.sum((coordinates - coordinates.iloc[index]) ** 2, axis=1))
        close_channel = dist_all.loc[dist_all < distance]

        # I'll extract the indexes of the the close channel (i'll exclude the channel with 0.0 distance)
        close_indexes = close_channel.loc[close_channel.index != index].index
        # Pick the current and neighbouring bad epochs vector
        neigh_bad_epochs = bad_trials[:, close_channel.index].sum(axis=1)

        # Consider just the good epoch (i.e. neigh_bad_epochs == 0)
        neigh_good_epochs = [x for x, v in enumerate(neigh_bad_epochs) if v == 0]
        channel_subgroup = data_channel[neigh_good_epochs, :, :][:, close_indexes, :]

        # Check if there are no neighbour with the given distance
        if len(channel_subgroup) == 0:
            deviation = np.nan
            warnings.warn('There are no neighbours within the given distance value')
        else:

            deviation = np.mean(data_channel[neigh_good_epochs, index, :]) - np.mean(np.mean(channel_subgroup))

        feat_current.append(deviation)

    return feat_current


def feature_cor_neigh_3d(dataset, coord, distance=15):
    """
    Function will calculate the mean Pearson correlation coefficient of each channel against the neighboring channels
    This function need that Epoch object has the electrode coordinates of the channels

    Parameters
    ----------
    dataset: mne.Epoch
        Data
    coord: pondas.Dataframe
        Dataframe containing the 3D coordinates of the channels
    distance: int
        Distance in mm used to extract the neighbouring channels


    Returns
    -------
    feat_current: list
        List of the mean Pearson correlation coefficients

    """

    # Extract the usable information from 'dataset'
    data_channel = dataset.get_data()
    channels_name = dataset.ch_names
    bad_trials = dataset.bad_trials

    # Initialization
    feat_current = []

    # Metadata info
    meta = coord.loc[coord['name'] == channels_name]

    # Retrieving the coordinate
    coordinates = meta[['x', 'y', 'z']]
    # Re-organize the indexes in order to match later with 'data_channel'
    coordinates = coordinates.reset_index(drop=True)

    for index, ch in enumerate(channels_name):
        dist_all = np.sqrt(np.sum((coordinates - coordinates.iloc[index]) ** 2, axis=1))
        close_channel = dist_all.loc[dist_all < distance]

        # I'll extract the indexes of the the close channel (i'll exclude the channel with 0.0 distance)
        close_indexes = close_channel.loc[close_channel.index != index].index
        # Pick the current and neighbouring bad epochs vector
        neigh_bad_epochs = bad_trials[:, close_channel.index].sum(axis=1)

        # Consider just the good epoch (i.e. neigh_bad_epochs == 0)
        neigh_good_epochs = [x for x, v in enumerate(neigh_bad_epochs) if v == 0]
        channel_subgroup = data_channel[neigh_good_epochs, :, :][:, close_indexes, :]

        # Rearrange dimension
        channel_subgroup = channel_subgroup.transpose(1, 2, 0)
        channel_subgroup = np.reshape(channel_subgroup, (channel_subgroup.shape[0], channel_subgroup.shape[1] * channel_subgroup.shape[2]), order='F')

        # Check if there are no neighbour with the given distance
        if len(channel_subgroup) == 0:
            correlation = np.nan
            warnings.warn('There are no neighbours within the given distance value')
        else:
            correlation = np.sum(np.corrcoef(data_channel[neigh_good_epochs, index, :].flatten(), channel_subgroup)[0, 1:]) / len(channel_subgroup)

        feat_current.append(correlation)

    return feat_current

