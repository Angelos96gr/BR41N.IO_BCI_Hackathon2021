# BR41N.IO_BCI_Hackathon2021

This repository contains the code developed in the BR41N.IO Brain Computer Interface Hackathon 2021 by Pölikübs consisting of Angelos Theocharis, Chiara Lambranzi and Alessandro Murari to receive 2nd place on the data analysis project "Locked In Syndrome". 

In this project, we analyzed a vibro-tactile P300 BCI data-set from a patient with locked-in syndrom in order to optimize pre-processing, feature extraction and classification algorithms.


The scripts require the mne https://mne.tools/stable/install/mne_python.html and autoreject https://autoreject.github.io/ packages.


The proposed pipeline follows:

1. Preprocessing:
    * filtering raw EEG with bandpass FIR filter at 0.1-30 Hz
    * epoching with -100 to 600 ms and baseline correction
    * artifact correction and rejection using autoreject
    
2. Feature extraction:
    * proposed using PCA to automatically extract features from epoched data 
    * extracted hand-crafted features in the time, frequency and complexity domain
    
3. Classification:
    * implemented 3 widely used classifiers such as Multilayer Perceptron (MLP), K-Nearest Neighbor (KNN) and Naive Bayes (NB)
    * hyperparamter tuning performed using 5-fold crossvalidation
    * obtain an average classification rate across multiple runs
