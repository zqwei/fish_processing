import imageio
import numpy as np

def videoWrite(arr, fileName, fps=30, axis=0):
    if axis ==-1:
        arr = arr.transpose([2, 0, 1])
    arr_ = (arr - arr.min())/(arr.max()-arr.min())*255
    with imageio.get_writer(fileName, fps=fps) as writer:
        for im in arr_:
            writer.append_data(im)
