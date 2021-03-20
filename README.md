# BR41N.IO_BCI_Hackathon2021

This repository contains the code developed in the BR41N.IO Brain Computer Interface Hackathon 2021.

In this project, we analyzed a vibro-tactile P300 BCI data-set from a patient with locked-in syndrom in order to optimize pre-processing, feature extraction and classification algorithms.

The proposed pipeline consists of:

1. Preprocessing:
    * filtering raw EEG with bandpass FIR filter at 0.1-30 Hz
    * epoching with -100 to 600 ms and baseline correction
    * artifact correction and rejection using autoreject
    
2. Feature extraction:
    * proposed using PCA automatically extract features from epoched data 
    * extracted hand-crafted features in the time, frequency and complexity domain
    
3. Classification:
    * implemented 3 widely used classifiers such as Multilayer Perceptron (MLP), K-nearest neighbor (KNN) and Naive Bayes (NB)
    * hyperparamter tuning performed using 5-fold crossvalidation
    * obtain an average classification rate across multiple runs