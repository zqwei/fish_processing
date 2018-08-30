import numpy as np
import scipy.io as sio
import h5py

# for plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
sns.set_style("white")

from skimage import io
import imageio

import os, sys
fish_path = os.path.abspath(os.path.join('..'))
if fish_path not in sys.path:
    sys.path.append(fish_path)

from utils import getCameraInfo

folderName = '/groups/ahrens/ahrenslab/Takashi/toZiqiang/02212018Fish2-1/'

'''
h5f = h5py.File('*/TM0000000_CM0_CHN00.h5', 'r')
imgStack = h5f['default'] # z, x, y
plt.imshow(imgStack[0], cmap='gray')
plt.show()
'''

cameraInfo = getCameraInfo.getCameraInfo(folderName)
pixel_x0, pixel_x1, pixel_y0, pixel_y1 = [int(_) for _ in cameraInfo['camera_roi'].split('_')]
pixel_x = (pixel_x0, pixel_x1)
pixel_y = (pixel_y0, pixel_y1)

imgFileName = 'Raw_stack.tif'
imgFile = os.path.join(folderName, imgFileName)
imgStack = io.imread(imgFile)
# check the first image
plt.imshow(imgStack[0], cmap='gray')
plt.show()


####
### simple denoising
####
cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'
offset = np.load(cameraNoiseMat +'/offset_mat.npy')
gain = np.load(cameraNoiseMat +'/gain_mat.npy')
offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]

from pixelwiseDenoising.simpleDenioseTool import simpleDN
imgD = simpleDN(imgStack, offset=offset_, gain=gain_)
plt.imshow(imgD[0], cmap='gray')
# plt.colorbar()
plt.show()

####
### smooth dead pixels
####
from scipy.ndimage.filters import median_filter
win_ = 3
imgD_ = median_filter(imgD, size=(1, win_, win_))
plt.imshow(imgD.mean(axis=0))
plt.show()
# plt.figure(figsize=(20,10))
plt.imshow(imgD_.mean(axis=0))
plt.show()

####
### motion correction
####
from imageRegistration.imTrans import ImAffine
trans = ImAffine()
trans.level_iters = [1000, 1000, 100]
trans.ss_sigma_factor = 1.0
from utils.np_mp import parallel_to_chunks
t_ = len(imgStack)//2
win_ = 150
fix = imgStack[t_-win_:t_+win_].mean(axis=0)
fix_ = imgD_[t_-win_:t_+win_].mean(axis=0)

def regidStacks(move, fix=fix):
    if move.ndim < 3:
        move = move[np.newaxis, :]
    trans_move = move.copy()
    move_list = []
    for nframe, move_ in enumerate(move):
        trans_affine = trans.estimate_rigid2d(fix, move_)
        trans_mat = trans_affine.affine
        trans_move[nframe] = trans_affine.transform(move_)
        move_list.append([trans_mat[0, 1]/trans_mat[0, 0], trans_mat[0, 2], trans_mat[1, 2]])
    return trans_move, move_list

plt.imshow(fix)
plt.show()

imgStackMotion, imgStackMotionVar = parallel_to_chunks(regidStacks, imgStack, fix=fix)
imgDMotion, imgDMotionVar = parallel_to_chunks(regidStacks, imgD_, fix=fix_)

np.save('tmpData/imgStackMotion', imgStackMotion)
np.save('tmpData/imgStackMotionVar', imgStackMotionVar)
np.save('tmpData/imgDMotion', imgDMotion)
np.save('tmpData/imgDMotionVar', imgDMotionVar)


imgStackMotion = np.load('tmpData/imgStackMotion.npy')
imgDMotion = np.load('tmpData/imgDMotion.npy')
imgStackMotionVar = np.load('tmpData/imgStackMotionVar.npy')
imgDMotionVar = np.load('tmpData/imgDMotionVar.npy')

crop_x = (65, 90)
crop_y = (70, 100)
mask = np.zeros(fix.shape)
mask[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]] = 1
cropRaw = imgStackMotion[:, mask==1]
cropSDN = imgDMotion[:, mask==1]
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cropRaw[0].reshape(crop_x[1]-crop_x[0], crop_y[1]-crop_y[0]))
ax1.set_title('Raw image')
ax1.axis('off')
ax2.imshow(cropSDN[0].reshape(crop_x[1]-crop_x[0], crop_y[1]-crop_y[0]))
ax2.set_title('Denoised image')
ax2.axis('off')
plt.show()

title = [r'$\theta$', 'x', 'y']
f, ax = plt.subplots(1, 3, figsize = (25, 5))
for nplot in range(3):
    ax[nplot].plot(imgStackMotionVar[:, nplot]) # phase, x, y
    ax[nplot].plot(imgDMotionVar[:, nplot])
    ax[nplot].set_ylabel(r'$\Delta$ ' + title[nplot])
    ax[nplot].set_xlabel('Frame')
plt.show()


def compute_dff(sig):
    bg = np.percentile(sig, 5, axis=1)
    return (sig.mean(axis=1)-sig.mean(axis=1).mean())/(sig.mean(axis=1).mean()-bg)


dff_raw = compute_dff(cropRaw)
dff_sdn = compute_dff(cropSDN)

plt.plot(dff_raw, label='Raw data')
plt.plot(dff_sdn, label='Denoised data', alpha=0.5)
plt.ylabel('DF/F (pixel intensity)')
plt.xlabel('Time (frame)')
plt.legend()
plt.show()
