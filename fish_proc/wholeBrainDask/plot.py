from matplotlib import pyplot as plt

def plot_registration(trans_data, ref_img, step=10)
    for n in range(0, trans_data.shape[0], step):
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        ax[0].imshow(trans_data[n].max(0))
        ax[1].imshow(ref_img.max(0))
        ax[2].imshow(trans_data[n].max(0) - ref_img.max(0), vmax=20, vmin=-20)
        plt.show()
