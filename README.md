# Data processing
This is the data processing used for the calcium and voltage imaging

## Example data
`/nrs/ahrens/Ziqiang/Takashi_DRN_project/RawData/10182018/Fish3-2/`

## Environment setup
* Dependency:
    * [Dipy](http://nipy.org/dipy/) -- code for Registration

# Processing components
## Preprocessing of raw images -- Pixelwise denoising
`python DRN_processing_jobs.py pixel_denoise`
## Image registration and motion correction
`python DRN_processing_jobs.py registration`
## Cell segmentation
* Detrend: `python DRN_processing_jobs.py video_detrend`
* Denoise (local PCA): `python DRN_processing_jobs.py local_pca`
* Demix (local NMF, initialization using superpixels): `python DRN_processing_jobs.py demix_components` (default -- demix only using middle 1/3 of the time series)

## TODO
### Parallel processing
* Spark
* Parallel processing -- Dask
* GPU supports
* On-line version of NMF
* long time series data
