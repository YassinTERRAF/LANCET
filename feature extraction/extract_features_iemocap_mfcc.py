#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing (adapted)
# Date: 2022-03-07 (adapted)
# E-mail: zhuwenjing02@duxiaoman.com

import os
import glob
from tqdm import tqdm
import librosa
import numpy as np
import argparse
import pickle
import math
from collections import Counter
import random
import json
from python_speech_features import logfbank, fbank, sigproc
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed  # Import joblib

# Paths to datasets (original and modified versions)
datasets_path = ""
noisy_datasets_path = ""
output_dir = ""

class FeatureExtractor(object):
    def __init__(self, sample_rate, nmfcc=26):
        self.sample_rate = sample_rate
        self.nmfcc = nmfcc

    def get_features(self, features_to_use, X):
        X_features = None
        accepted_features_to_use = ("logfbank", 'mfcc', 'fbank', 'melspectrogram', 'spectrogram', 'interspeech2018')
        if features_to_use not in accepted_features_to_use:
            raise NotImplementedError(f"{features_to_use} not in {accepted_features_to_use}!")
        if features_to_use == 'logfbank':
            X_features = self.get_logfbank(X)
        elif features_to_use == 'mfcc':
            X_features = self.get_mfcc(X, self.nmfcc)
        elif features_to_use == 'fbank':
            X_features = self.get_fbank(X)
        elif features_to_use == 'melspectrogram':
            X_features = self.get_melspectrogram(X)
        elif features_to_use == 'spectrogram':
            X_features = self.get_spectrogram(X)
        elif features_to_use == 'interspeech2018':
            X_features = self.get_spectrogram_interspeech2018(X)
        return X_features

    def get_logfbank(self, X):
        def _get_logfbank(x):
            out = logfbank(signal=x, samplerate=self.sample_rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000, nfilt=40)
            return out

        X_features = np.apply_along_axis(_get_logfbank, 1, X)
        return X_features

    def get_mfcc(self, X, nmfcc=13):
        def _get_mfcc(x):
            mfcc_data = librosa.feature.mfcc(y=x, sr=self.sample_rate, n_mfcc=nmfcc)
            return mfcc_data

        X_features = np.apply_along_axis(_get_mfcc, 1, X)
        return X_features

    def get_fbank(self, X):
        def _get_fbank(x):
            out, _ = fbank(signal=x, samplerate=self.sample_rate, winlen=0.040, winstep=0.010, nfft=1024)
            return out

        X_features = np.apply_along_axis(_get_fbank, 1, X)
        return X_features

    def get_melspectrogram(self, X):
        def _get_melspectrogram(x):
            mel = librosa.feature.melspectrogram(y=x, sr=self.sample_rate, n_fft=800, hop_length=400)[np.newaxis, :]
            out = np.log10(mel).squeeze()
            return out

        X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
        return X_features

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features

    def get_spectrogram_interspeech2018(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.magspec(frames, NFFT=3198)
            out = out / out.max() * 2 - 1 
            out = np.sign(out) * np.log(1+255*np.abs(out))/np.log(256)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features

def segment(wavfile, sample_rate=16000, segment_length=2, overlap=1, padding=None):
    if isinstance(wavfile, str):
        wav_data, _ = librosa.load(wavfile, sr=sample_rate)
    elif isinstance(wavfile, np.ndarray):
        wav_data = wavfile
    else:
        raise ValueError(f'Type {type(wavfile)} is not supported.')
    
    X = []
    seg_wav_len = segment_length * sample_rate
    wav_len = len(wav_data)

    if seg_wav_len > wav_len: 
        if padding:
            n = math.ceil(seg_wav_len / wav_len)
            wav_data = np.hstack(n * [wav_data])
        else:
            return None, None
    
    index = 0
    while (index + seg_wav_len <= wav_len):
        X.append(wav_data[int(index):int(index + seg_wav_len)])
        assert segment_length - overlap > 0
        index += int((segment_length - overlap) * sample_rate)

    X = np.array(X)
    return X

def process(wavfiles, labels, num_label=None, features_to_use='mfcc', sample_rate=16000, nmfcc=26, 
            train_overlap=1, test_overlap=1.6, segment_length=2, split_rate=0.8, 
            featuresFileName='features.pkl', toSaveFeatures=True, aug=None, padding=None, n_jobs=-1):

    # Perform stratified split to ensure balance of classes in each set
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        wavfiles, labels, test_size=0.2, stratify=labels, random_state=42)

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

    get_label = lambda x: x[1]
    train_info = json.dumps(Counter(train_labels))
    val_info = json.dumps(Counter(val_labels))
    test_info = json.dumps(Counter(test_labels))
    info = {'train': train_info, 'val': val_info, 'test': test_info}

    if num_label is not None:
        print(f'Amount of categories: {num_label}')
    print(f'Training Datasets: {len(train_files)}, Validation Datasets: {len(val_files)}, Testing Datasets: {len(test_files)}')

    if aug == 'upsampling':
        label_wav = {
            'neutral': [],
            'happy': [],
            'sad': [],
            'angry': [],
        }
        for wavfile, label in zip(train_files, train_labels):
            label_wav[label].append(wavfile)
        maxval = max(len(w) for w in label_wav.values())
        for l, w in label_wav.items():
            nw = len(w)
            indices = list(np.random.choice(range(nw), maxval - nw, replace=True))
            for i in indices:
                train_files.append(w[i])
                train_labels.append(l)
        random.shuffle(list(zip(train_files, train_labels)))
        print(f'After Augmentation...\nTraining Datasets: {len(train_files)}, Validation Datasets: {len(val_files)}, Testing Datasets: {len(test_files)}')

    feature_extractor = FeatureExtractor(sample_rate, nmfcc)

    print('Extracting features for training datasets')

    # Parallelize feature extraction using joblib
    def extract_features(wavfile, label):
        X1 = segment(wavfile, sample_rate=sample_rate, segment_length=segment_length, overlap=train_overlap, padding=padding)
        if X1 is None:
            return None, None
        X1 = feature_extractor.get_features(features_to_use, X1)
        return X1, label

    # Use joblib Parallel to process in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(extract_features)(wavfile, label) for wavfile, label in tqdm(zip(train_files, train_labels), total=len(train_files)))

    train_X, train_y = [], []
    for X1, label in results:
        if X1 is not None:
            train_X.append(X1)
            train_y.extend([label] * len(X1))
    
    train_X = np.row_stack(train_X)
    train_y = np.array(train_y)

    print(f'Amount of categories after segmentation(training): {Counter(train_y).items()}')
    assert len(train_X) == len(train_y), "X length and y length must match!"

    print('Extracting features for validation datasets')

    results = Parallel(n_jobs=n_jobs)(delayed(extract_features)(wavfile, label) for wavfile, label in tqdm(zip(val_files, val_labels), total=len(val_files)))

    val_dict = []
    val_y = []
    for X1, label in results:
        if X1 is not None:
            val_dict.append({'X': X1, 'y': label})
            val_y.extend([label] * len(X1))
    print(f'Amount of categories after segmentation(validation): {Counter(val_y).items()}')

    print('Extracting features for test datasets')

    results = Parallel(n_jobs=n_jobs)(delayed(extract_features)(wavfile, label) for wavfile, label in tqdm(zip(test_files, test_labels), total=len(test_files)))

    test_dict = []
    test_y = []
    for X1, label in results:
        if X1 is not None:
            test_dict.append({'X': X1, 'y': label})
            test_y.extend([label] * len(X1))
    print(f'Amount of categories after segmentation(test): {Counter(test_y).items()}')

    info['train_seg'] = f'{Counter(train_y).items()}'
    if toSaveFeatures:
        print(f'Saving features to {featuresFileName}.')
        features = {'train_X': train_X, 'train_y': train_y, 'val_dict': val_dict, 'test_dict': test_dict, 'info': info}
        with open(featuresFileName, 'wb') as f:
            pickle.dump(features, f)

    return train_X, train_y, val_dict, test_dict, info

def process_IEMOCAP(datasets_path, LABEL_DICT, datadir='data/', featuresFileName=None, features_to_use='mfcc', 
                    impro_or_script='impro', sample_rate=16000, nmfcc=26, train_overlap=1, test_overlap=1.6, 
                    segment_length=2, split_rate=0.8, toSaveFeatures=True, aug=None, padding=None, **kwargs):

    if not os.path.exists(datadir):
        os.makedirs(datadir)
    num_label = {}
    if featuresFileName is None:
        featuresFileName = f'{datadir}/features_{features_to_use}_{impro_or_script}.pkl'
    
    if os.path.exists(datasets_path):
        wavdirname = datasets_path + '/*/sentences/wav/*/S*.wav'
        allfiles = glob.glob(wavdirname)
    else:
        raise ValueError(f'{datasets_path} not existed.')

    wavfiles, labels = [], []
    for wavfile in allfiles:
        if len(os.path.basename(wavfile).split('-')) < 5: 
            continue
        label = str(os.path.basename(wavfile).split('-')[2])
        if label not in LABEL_DICT: 
            continue
        if impro_or_script != 'all' and (impro_or_script not in wavfile): 
            continue
        wav_data, _ = librosa.load(wavfile, sr=sample_rate)
        seg_wav_len = segment_length * sample_rate
        wav_len = len(wav_data)
        if seg_wav_len > wav_len:
            if padding:
                n = math.ceil(seg_wav_len / wav_len)
                wav_data = np.hstack(n * [wav_data])
            else:
                continue

        label = LABEL_DICT[label]
        wavfiles.append(wav_data)
        labels.append(label)
        num_label[label] = num_label.get(label, 0) + 1

    train_X, train_y, val_dict, test_dict, info = process(
        wavfiles, labels, num_label=num_label, features_to_use=features_to_use,
        sample_rate=sample_rate, nmfcc=nmfcc, train_overlap=train_overlap, 
        test_overlap=test_overlap, segment_length=segment_length, 
        split_rate=split_rate, featuresFileName=featuresFileName, 
        toSaveFeatures=toSaveFeatures, aug=aug, padding=padding)
    
    return train_X, train_y, val_dict, test_dict, info

IEMOCAP_LABEL = {
    '01': 0,  # neutral
    '04': 1,  # sad
    '05': 2,  # angry
    '07': 3,  # happy/excitement merged
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing datasets')
    parser.add_argument('-d', '--datasets_path', default=datasets_path, type=str, help='Path to clean datasets')
    parser.add_argument('--noisy_datasets_path', default=noisy_datasets_path, type=str, help='Path to noisy datasets')
    parser.add_argument('--datadir', default=output_dir, type=str, help='Directory to save extracted features')
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of parallel jobs')

    args = parser.parse_args()

    # Process and save features for each dataset type
    for condition_name in ["clean", "babble", "noise", "music", "white", "reverberation"]:

        if condition_name == "clean":
            clean_data_dir = os.path.join(args.datadir, "clean")
            if not os.path.exists(clean_data_dir):
                os.makedirs(clean_data_dir)

            process_IEMOCAP(
                datasets_path=args.datasets_path,
                LABEL_DICT=IEMOCAP_LABEL,
                datadir=clean_data_dir,
                features_to_use='mfcc',
                impro_or_script='impro',
                sample_rate=16000, 
                nmfcc=26,
                train_overlap=1, 
                test_overlap=1.6, 
                segment_length=2,
                split_rate=0.8,
                toSaveFeatures=True,
                aug=None,
                padding=None,
                n_jobs=args.n_jobs
            )
            
        elif condition_name in ["babble", "noise", "music", "white"]:
            for snr in ["-5", "0", "5", "10", "15", "20"]:
                dataset_path = f"{args.noisy_datasets_path}/{condition_name}/{snr}"
                condition_dir = f"{args.datadir}/{condition_name}/{snr}"
                process_IEMOCAP(
                    datasets_path=dataset_path,
                    LABEL_DICT=IEMOCAP_LABEL,
                    datadir=condition_dir,
                    features_to_use='mfcc',
                    impro_or_script='impro',
                    sample_rate=16000, 
                    nmfcc=26,
                    train_overlap=1, 
                    test_overlap=1.6, 
                    segment_length=2,
                    split_rate=0.8,
                    toSaveFeatures=True,
                    aug=None,
                    padding=None,
                    n_jobs=args.n_jobs
                )
