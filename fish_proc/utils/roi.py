from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, img):
        self.canvas = ax.figure.canvas
        self.ax = ax
        dx, dy = img.shape
        xv, yv = np.meshgrid(np.arange(dx),np.arange(dy), indexing='ij')
        self.pix = np.vstack((yv.flatten(),xv.flatten())).T
        self.array = np.zeros((dx, dy))
        self.msk = ax.imshow(self.array, vmin=0, vmax=1, cmap='Blues', alpha=0.5)
        lineprops = {'color': 'red', 'linewidth': 4, 'alpha': 0.4}
        self.lasso = LassoSelector(ax, onselect=self.onselect, lineprops=lineprops, useblit=False)
        self.ind = []
        self.verts=[]
        self.array_list=[]

    def onselect(self, verts):
        def updateArray(array, indices):
            lin = np.arange(array.size)
            newArray = array.flatten()
            newArray[lin[indices]] = 1
            return newArray.reshape(array.shape)
        p = Path(verts)
        self.verts.append(verts)
        self.ind = p.contains_points(self.pix, radius=1)
        tmp = updateArray(self.array, self.ind)
        self.array_list.append(tmp)
        x, y = np.array(verts).T
        self.ax.plot(x, y, '-', lw=2)
        self.msk.set_data(tmp)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()