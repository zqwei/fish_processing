'''
Copyright 2018 Wei, Ziqiang, Janelia Research Campus

weiz@janelia.hhmi.org
'''

import numpy as np
import deprecation

def roll_scale(x, win_=50001, factor=1):
    import pandas as pd
    x = pd.Series(x)
    x_ave = x.rolling(window=win_, center=True).median()
    x_ave[0]=x_ave[x_ave.first_valid_index()]
    x_ave = x_ave.astype(float).interpolate(method='linear')
    x_std = x.rolling(window=win_, center=True).std()
    x_std[0]=x_std[x_std.first_valid_index()]
    x_std = x_std.astype(float).interpolate(method='linear')
    return ((x-x_ave)/x_std/factor).values
    

def mad_scale(x, axis=0, c=0.6745):
    from statsmodels.robust.scale import mad
    # from scipy.stats import norm as Gaussian
    # c = Gaussian.ppf(3/4.)
    return (x-np.median(x, axis=axis))/mad(x, c=c, axis=axis, center=np.median)

def cluster_spikes(spkc, spkprob, voltr, spk=None, print_en=False, plot_en=False):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    spk1 = spkc.copy()
    spk2 = spkc.copy()
    data = np.vstack((voltr, spkprob)).T
    data = data[spkc==1, :]
    data = StandardScaler().fit_transform(data)
    is_conv = False
    eps = 0.5
    
    while not is_conv:
        db = DBSCAN(eps=eps, min_samples=10).fit(data)
        labels = db.labels_
        if labels.max()==0:
            is_conv = True
            continue
        if labels.max()==1 and labels.min()==0:
            is_conv = True
            continue
        eps += 0.1
    
    if plot_en:
        for nlabel in range(-1, labels.max()+1):
            plt.scatter(data[labels==nlabel, 0], data[labels==nlabel, 1])
        plt.show()
        
    if labels.max()==1:
        if voltr[spkc==1][labels==0].mean()>voltr[spkc==1][labels==1].mean():
            labels = 1-labels
        spk1[spkc==1] = labels==1
        spk2[spkc==1] = labels==0
    else:
        spk2[:] = False
    
    if print_en and spk is not None:
        print('-------------------')
        print('Type #1 spike')
        print_spike_detection_report(spk, spk1, None)
        if spk2.sum()>0:
            print('-------------------')
            print('Type #2 spike')
            print_spike_detection_report(spk, spk2, None)
    return spk1, spk2    
        

def plot_spks(plt, spkcount, ratio, label='Raw spike time'):
    spkTime = spkcount>0
    plt.plot(np.array(np.where(spkTime)).T, spkcount[spkTime]*ratio,'+', label=label)


def plot_test_performance(m, x_test, labels, plt):
    import seaborn as sns
    pred_x_test = m.predict(x_test)
    labels = labels.astype(np.bool)
    sns.distplot(pred_x_test[labels], label='Spike')
    sns.distplot(pred_x_test[~labels], label='None-Spike')
    sns.despine()
    plt.xlim([0,1])
    plt.xlabel('Predition of spike probability')
    plt.ylabel('Counts')
    plt.legend()


def detected_window_max_spike(pred_x, voltr, window_length = 41, peak_wid=2, thres=0.5):
    spkInWindow = pred_x>thres
    spkInWindow = spkInWindow.flatten()
    hwin = window_length//2
    # spkInWindow[:window_length] = False
    spkcount = np.zeros(voltr.shape[0]).astype(np.bool)
    spkprob = np.zeros(voltr.shape[0])
    for idx, nspk in enumerate(spkInWindow):
        if nspk:
            x = voltr[idx+hwin-peak_wid:idx+hwin+peak_wid+1]
            spkcount[idx+hwin-peak_wid+np.argmax(x)] = True # first max is set to be spike time
            spkprob[idx+hwin-peak_wid+np.argmax(x)]=pred_x[idx]
    # remove neighbouring spikes ---
    spkcount[np.where(np.logical_and(spkcount[:-1], spkcount[1:]))[0]+1] = False
    return spkcount, spkprob


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
    if title_string is not None:
        print(title_string)
    print('Total ephys spikes %d'%(totspk))
    print('Total detected spikes %d'%(totspk_))
    print('Found spikes %d'%(TPspk))
    print('Unfound spikes %d'%(totspk - TPspk))
    print('Extra spikes %d'%(totspk_ - TPspk))


@deprecation.deprecated(details="Use detected_window_max_spike function instead")
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
    return spkcount_

@deprecation.deprecated(details="Use detected_window_max_spike function instead")
def detected_peak_spikes(m, x_, voltr_, thres=0.4, devoltr_ = None, peakThres=.9, peak_minDist=10, smallPeakThres = 20):
    import peakutils
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

@deprecation.deprecated(details="Use detected_window_max_spike function instead")
def _detected_window_max_spike(m, x_, voltr_, peak_wid=2, thres=0.5):
    pred_x_test = m.predict(x_)
    spkInWindow = pred_x_test>thres
    spkInWindow = spkInWindow.flatten()
    window_length = x_.shape[1]
    hwin = window_length//2
    spkInWindow[:window_length] = False
    spkcount_ = np.zeros(voltr_.shape[0]).astype(np.bool)
    for idx, nspk in enumerate(spkInWindow):
        if nspk:
            x__ = voltr_[idx+hwin-peak_wid:idx+hwin+peak_wid+1]
            spkcount_[idx+hwin-peak_wid+np.argmax(x__)] = True # first max is set to be spike time
    # remove neighbouring spikes ---
    spkcount_[np.where(np.logical_and(spkcount_[:-1], spkcount_[1:]))[0]+1] = False
    return spkcount_