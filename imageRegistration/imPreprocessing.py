def downscale(v, scale_factors):
    from skimage.transform import downscale_local_mean
    from skimage.exposure import rescale_intensity
    from scipy.stats import zscore
    return rescale_intensity(zscore(downscale_local_mean(v, scale_factors), axis=None), out_range=(0,1))
