import imageio
import numpy as np

def videoWrite(arr, fileName, fps=30):
    arr_ = (arr - arr.min())/(arr.max()-arr.min())*255
    with imageio.get_writer(fileName, fps=fps) as writer:
        for im in arr_:
            writer.append_data(im)
