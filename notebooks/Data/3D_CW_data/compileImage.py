import numpy as np
from skimage import io
from glob import glob
import os

folderName = '17-08-24-L2-CL-Brain-Raw'

tifStack = []
for file_ in sorted(glob(folderName+'/*.tif')):
    tifStack.append(io.imread(file_))

tifStack = np.array(tifStack)

np.save(folderName, tifStack)
