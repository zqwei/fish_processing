import imageio
from skimage.exposure import rescale_intensity as rescale
import cv2
import numpy as np

def videoWrite(arr, fileName, fps=30, axis=0):
    if axis ==-1:
        arr = arr.transpose([2, 0, 1])
#     arr_ = (arr - arr.min())/(arr.max()-arr.min())*255
#     arr_ = arr_.astype('uint8')
    arr = rescale(arr, out_range='uint8').astype('uint8')
    with imageio.get_writer(fileName, fps=fps) as writer:
        for im in arr:
            writer.append_data(im)

def writeVideoFile(images, nfile_name, frameRate = 30):
    # using opencv to write a movie
    tlen, ImageWidth, ImageHeight = images.shape
    writer = cv2.VideoWriter(nfile_name + '.MJPEG', cv2.VideoWriter_fourcc(*'MJPG'), frameRate, (ImageWidth, ImageHeight), False)
    images = (images - images.min())/(images.max() - images.min())*255
    images = images.astype('uint8')
    for img in images:
        writer.write(img)
    writer.release()

