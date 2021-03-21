import scipy.io as sio
import matplotlib.pyplot as plt
import os.path as op
import glob
import numpy as np
import pandas as pd

# modules used for data preprocessing
from epoching import load_data, find_trigs, make_epo, autore

# modules used for feature extraction
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from feature_extraction import extract

# modules used for classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# used for keeping track of classification performance
acc_MLP  = []
acc_NB   = []
acc_KNN  = []

# number of runs to get the average classification performance
num_runs = 5

#plt.close('all')
for i in range(num_runs):
    # define the directory where the data is stred
    dir_eeg = 'D:\\brainHack\\branIOHackathon2021\data'

    # read all eeg files in this directory
    file_names = glob.glob(op.join(dir_eeg, f'**.mat'))

    # the first file used for training, the second for testing
    f_tr = file_names[0]
    f_tst = file_names[1]


    # read data matrix and extract relevant information
    data_tr = sio.loadmat(f_tr)
    data_tst = sio.loadmat(f_tst)
    y, trig, fs, time_vect = load_data(data_tr)
    y_tst, trig_tst, fs, time_vect = load_data(data_tst)
    [idx_n1, idx_p1, idx_p2] = find_trigs(trig)
    [idx_n1_tst, idx_p1_tst, idx_p2_tst] = find_trigs(trig_tst)

    ########### Preprocessing ###########

    # make epochs using mne package according to trigger type
    epo_eeg_p1_tr = make_epo(y / 10e6, idx_p1, trig)
    epo_eeg_p2_tr = make_epo(y / 10e6, idx_p2, trig)

    epo_eeg_p1_tst = make_epo(y_tst / 10e6, idx_p1_tst, trig_tst)
    epo_eeg_p2_tst = make_epo(y_tst / 10e6, idx_p2_tst, trig_tst)

    # artifact correction, rejection with autoreject
    epo_eeg_p1_tr_cl = autore(epo_eeg_p1_tr)
    epo_eeg_p2_tr_cl = autore(epo_eeg_p2_tr)

    epo_eeg_p1_tst_cl = autore(epo_eeg_p1_tst)
    epo_eeg_p2_tst_cl = autore(epo_eeg_p2_tst)

    '''
    ######## Plotting section ############
    
    fig, axes = plt.subplots(2,2, sharex = True, figsize=(20, 10))
    fig.suptitle(f_tr[f_tr.find('P'):f_tr.find('.')], fontsize=12)
    epo_eeg_p1_tr.average().plot(axes = axes[0,0],titles = "+1 trigger before preprocessing",spatial_colors = True)
    epo_eeg_p2_tr.average().plot(axes = axes[0,1],titles = "+2 trigger before preprocessing",spatial_colors = True)
    epo_eeg_p1_tr_cl.average().plot(axes = axes[1,0],titles = "+1 trigger after autoreject",spatial_colors = True)
    epo_eeg_p2_tr_cl.average().plot(axes = axes[1,1],titles = "+2 trigger after autoreject",spatial_colors = True)
    
    '''

    ########### Feature Extraction ###########

    # preparation and feature extraction of training set
    Datmat = np.concatenate((epo_eeg_p1_tr_cl._data, epo_eeg_p2_tr_cl._data))
    labels = np.concatenate((epo_eeg_p1_tr_cl.events[:, 2], epo_eeg_p2_tr_cl.events[:, 2]))
    epoch = Datmat.transpose(2, 0, 1)
    df = pd.DataFrame()

    for ch in range(epoch.shape[2]):
        feature = extract(epoch[:, :, ch], fs, 0.1, amplitude=True, amplitude_P300=True,
                          kurtosis=True, skewness=True, std=True, sampen=True, rms=True, hurst=True, gradient=True,
                          alfa=True, beta=True, theta=True, delta=True, broad_band=True,)

        current = pd.DataFrame(feature)
        current['class'] = labels - 1

        df = pd.concat([df, current], ignore_index=True)



    minmax = MinMaxScaler().fit(df.iloc[:, :-1].values)
    data_stand = minmax.transform(df.iloc[:, :-1].values)

    pca = PCA(n_components=0.99).fit(data_stand)
    df_pca = pd.DataFrame(pca.transform(data_stand))


    # preparation and feature extraction of testing set

    Datmat_tst = np.concatenate((epo_eeg_p1_tst_cl._data, epo_eeg_p2_tst_cl._data))
    labels_tst = np.concatenate((epo_eeg_p1_tst_cl.events[:, 2], epo_eeg_p2_tst_cl.events[:, 2]))
    epoch_tst = Datmat_tst.transpose(2, 0, 1)
    df_tst = pd.DataFrame()

    for ch in range(epoch.shape[2]):
        feature_tst = extract(epoch_tst[:, :, ch], fs, 0.1, amplitude=True, amplitude_P300=True,
                              kurtosis=True, skewness=True, std=True, sampen=True, rms=True, hurst=True, gradient=True,
                              alfa=True, beta=True, theta=True, delta=True,
                              base_theta=False, broad_band=True, base_high_gamma=False)



        current_tst = pd.DataFrame(feature_tst)
        current_tst['class'] = labels_tst - 1

        df_tst = pd.concat([df_tst, current_tst], ignore_index=True)

    # if we want to apply standardization and pca on the test set
    # Data_stand_tst = minmax.transform(Data_tst)
    # Data_pca_tst = pca.transform(Data_stand_tst)



    ########### Classification ###########
    feats_train, feats_test, label_train, label_test = train_test_split(df.iloc[:, 0:-1].values, df['class'], test_size=0.3,
                                                                        stratify=df['class'], random_state= 77)

    # define classifiers
    mlp_original = MLPClassifier()
    gnb = GaussianNB()
    neigh = KNeighborsClassifier(n_neighbors=3)


    # define hyperparameter searching with crossvalidation
    opt_parameters = {"hidden_layer_sizes": [(8, 4)], "activation": ["relu"], "solver": ["adam"], 'batch_size': [50],
                      "max_iter": [1000], "alpha": [0.5]}
    mlp_CV = GridSearchCV(mlp_original, opt_parameters, cv=5, scoring="accuracy")
    mlp = mlp_CV.fit(feats_train, label_train).best_estimator_

    # performance of mlp on training set and test set using MLP
    label_train_pred = mlp.predict(feats_train)
    label_test_pred = mlp.predict(feats_test)
    label_real_test_pred = mlp.predict(df_tst.iloc[:, :-1].values)

    # performance of mlp on training set and test set using NB
    parameters_nb = {}
    gs_nb = GridSearchCV(gnb, parameters_nb, cv=5, scoring='accuracy', verbose=1, n_jobs=4, refit=True)
    gs_nb.fit(feats_train, label_train)
    y_pred_NB = gs_nb.predict(df_tst.iloc[:, 0:-1].values)

    # performance of mlp on training set and test set using KNN
    neigh.fit(df.iloc[:, 0:-1].values, df.iloc[:, -1].values)
    y_pred_KNN = neigh.predict(df_tst.iloc[:, 0:-1].values)

    # keep track of mean accuracy
    #sum_MLP = sum_MLP + accuracy_score(df_tst.iloc[:, -1].values, label_real_test_pred)
    acc_MLP.append(accuracy_score(df_tst.iloc[:, -1].values, label_real_test_pred))
    #sum_NB = sum_NB + accuracy_score(df_tst.iloc[:, -1].values, y_pred_NB)
    acc_NB.append(accuracy_score(df_tst.iloc[:, -1].values, y_pred_NB))
    #sum_KNN = sum_KNN + accuracy_score(df_tst.iloc[:, -1].values, y_pred_KNN)
    acc_KNN.append(accuracy_score(df_tst.iloc[:, -1].values, y_pred_KNN))


print('**** Prediction Results over %d runs****' %num_runs)
print("Average performance of MLP: %.2f" %np.mean(acc_MLP))
print("Average performance of NB: %.2f" %np.mean(acc_NB))
print("Average performance of KNN: %.2f" %np.mean(acc_KNN))
