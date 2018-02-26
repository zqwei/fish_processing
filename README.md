# Fish data processing

## 0. Environment setup

### 0.1 Reference to Davis' flow of spark setting (bash file etc.)

#### 0.1.1 Set up parallel processing on local or remote machine

### 0.2 Basic python environment (jupyter notebook)
```python
%matplotlib inline
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
```
#### 0.2.1 Load image files
```python
h5f = h5py.File('*/TM0000000_CM0_CHN00.h5', 'r')
imgStack = h5f['default'] # z, x, y
plt.imshow(imgStack[0], cmap='gray')
plt.show()
```

## 1. Preprocessing of raw images -- Pixelwise denoising

## 2. Image registration and motion correction

### 2.1 Registration to a single fish with single modularity (motion correction)

### 2.2 Registration to a single fish with multiple modularities from the same fish

#### 2.2.1 Registration of a single fish to a well-known brain atlas

### 2.3 Average registration across fishes

## 3. Single-cell level denoising

## 4. Brain-wide analyses

### 4.1 Spatial components

### 4.2 Temporal components and oscillations

### 4.3 Spatiotemporal analyses and dynamical systems

### 4.4 Brain states and change detections

## 5. Behavioral data

## Extra 1. Single-cell RNA seq
https://github.com/zqwei/single_cell_zebrafish
