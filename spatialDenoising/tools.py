import numpy as np
import scipy as sp

def remove_trend(Y_rm,detrend_option='linear'):
    mean_pixel = Y_rm.mean(axis=1, keepdims=True)
    Y_rm2 = Y_rm - mean_pixel
    # Detrend
    if detrend_option=='linear':
        detr_data = sp.signal.detrend(Y_rm2,axis=1,type='l')
    #elif detrend_option=='quad':
        #detr_data = detrend(Y_rm)
    else:
        print('Add option')
    Y_det = detr_data + mean_pixel
    offset = Y_rm - Y_det
    return Y_det, offset


