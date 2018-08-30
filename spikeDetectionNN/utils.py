'''
Copyright 2018 Wei, Ziqiang, Janelia Research Campus

weiz@janelia.hhmi.org
'''

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("ticks")
import numpy as np
import peakutils


def plot_spks(plt, spkcount, ratio, label='Raw spike time'):
    spkTime = spkcount>0
    plt.plot(np.array(np.where(spkTime)).T, spkcount[spkTime]*ratio,'o', label=label)


def plot_test_performance(m, x_test, labels, plt=plt):
    pred_x_test = m.predict(x_test)
    labels = labels.astype(np.bool)
    sns.distplot(pred_x_test[labels], label='Spike')
    sns.distplot(pred_x_test[~labels], label='None-Spike')
    sns.despine()
    plt.xlim([0,1])
    plt.xlabel('Predition of spike probability')
    plt.ylabel('Counts')
    plt.legend()

def detected_max_spike(m, x_, voltr_, thres=0.4):
    pred_x_test = m.predict(x_)
    window_length = x_.shape[1]
    spkInWindow = pred_x_test>thres
    spkInWindow = spkInWindow.flatten()
    spkInWindow[:window_length] = False
    spkcount_ = np.zeros(voltr_.shape[0]).astype(np.bool)
    for idx, nspk in enumerate(spkInWindow):
        if nspk:
            x__ = voltr_[idx:idx+window_length]
            spk_idx = np.where(x__ == x__.max()).squeeze()[0]
            spkcount_[idx+spk_idx] = True
    # diff = voltr_set - devoltr_set
    # diff_ = diff[spkcount_.astype(np.bool)]
    # thres = np.percentile(diff_, 20)
    # spk_diff = diff > thres
    # spkcount__ = np.logical_and(spkcount_, spk_diff)
    return spkcount_

def detected_peak_spikes(m, x_, voltr_, thres=0.4, devoltr_ = None, peakThres=.9, peak_minDist=10, smallPeakThres = 20):
    pred_x_test = m.predict(x_)
    window_length = x_.shape[1]
    spkInWindow = pred_x_test>thres
    spkInWindow = spkInWindow.flatten()
    spkInWindow[:window_length] = False
    spkcount_ = np.zeros(voltr_.shape[0]).astype(np.bool)
    for idx, nspk in enumerate(spkInWindow):
        if nspk:
            x__ = voltr_[idx:idx+window_length]
            spk_idx = peakutils.indexes(x__, thres=peakThres, min_dist=peak_minDist)
            spkcount_[idx+spk_idx] = True
    if (devoltr_ is not None) and spkcount_.sum()>0:
        diff = voltr_ - devoltr_
        diff_ = diff[spkcount_.astype(np.bool)]
        thres = np.percentile(diff_, smallPeakThres)
        spk_diff = diff > thres
        spkcount_ = np.logical_and(spkcount_, spk_diff)
    return spkcount_

def print_spike_detection_report(spk_in_range, spkc_in_range, title_string):
    matched = spk_in_range == spkc_in_range
    tot_ = len(spk_in_range)
    totspk = spk_in_range.sum()
    totspk_ = spkc_in_range.sum()

    matched_ = (matched).sum()
    idx = np.array(np.where(~matched)).flatten()
    idx_ = idx[1:] - idx[:-1]
    match1_ = (idx_==1).sum()*2
    match5_ = np.logical_and(idx_>1, idx_<6).sum()*2

    TPspk = 0
    winSize = 3
    for nspk in np.array(np.where(spk_in_range)).flatten():
        if spkc_in_range[max(0, nspk-winSize):min(nspk+winSize, len(spkc_in_range))].sum()>0:
            TPspk +=1
    print('-------------------')
    print(title_string)
    print('Total ephys spikes %d'%(totspk))
    print('Total detected spikes %d'%(totspk_))
    print('Found spikes %d'%(TPspk))
    print('Unfound spikes %d'%(totspk - TPspk))
    print('Extra spikes %d'%(totspk_ - TPspk))
