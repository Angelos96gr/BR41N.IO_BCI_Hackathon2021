from autoreject import AutoReject
import mne
import numpy as np


def load_data(data):
    """

    Parameters
    ----------
    data: Matlab
        structure

    Returns
    -------
    y: data
    trig: triggering label
    """
    y = data['y']
    trig = data['trig']
    fs = int(data['fs'])
    time_vect = np.linspace(0, y.shape[0] / fs, y.shape[0])

    return y, trig, fs, time_vect



def find_trigs(trig):
    idx = np.where(trig.squeeze() != 0)[0]  # Index of triggering
    idx_n1 = np.where(trig.squeeze() == -1)[0]  # Triggers for -1
    idx_p1 = np.where(trig.squeeze() == 1)[0]  # Triggers for +1
    idx_p2 = np.where(trig.squeeze() == 2)[0]  # Triggers for +2
    indic = [idx_n1, idx_p1, idx_p2]
    return indic


def make_epo(y, idx_cust, trig):
    fs = 256
    ch_names = ['C4', 'C3', 'Fz', 'Cz', 'CP1', 'CPz', 'CP2', 'Pz']
    ch_types = ['eeg'] * 8
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=fs)
    info.set_montage('standard_1020')

    raw_eeg = mne.io.RawArray(y.T, info)
    raw_eeg_filt = raw_eeg.copy().filter(0.1, 30)

    idx_cust = np.reshape(idx_cust, (idx_cust.shape[0], 1))
    event_id = np.reshape(trig[idx_cust, 0], (idx_cust.shape[0], 1))
    events = np.hstack((idx_cust, np.zeros((event_id.shape[0], 1)), event_id))

    epo_eeg = mne.Epochs(raw_eeg_filt, events=events.astype(int), tmin=-0.1, tmax=0.6, baseline=(None, 0), preload=True)
    return epo_eeg

def autore(epo_eeg_cust):

    ar = AutoReject(n_jobs=4)
    ar.fit(epo_eeg_cust)
    epo_ar, reject_log = ar.transform(epo_eeg_cust, return_log=True)
    clean = epo_ar.copy()
    # Used for plotting
    #scalings = dict(eeg=50)
    # reject_log.plot_epochs(epo_eeg_cust, scalings=scalings)
    # epo_ar.average().plot()
    return clean