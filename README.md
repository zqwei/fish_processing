# Data processing
This is the data processing used for the calcium and voltage imaging

## Environment setup
* Spark
* Parallel processing -- Dask
* Optional packages:
    * [Caiman](https://github.com/flatironinstitute/CaImAn) -- code for visualization, basic statistics such as SNR, local correlation etc.
    * [FunImag](https://github.com/paninski-lab/funimag) -- superpixel cell segmentation
    * [Trefide](https://github.com/ikinsella/trefide) -- trend filtering, total variational denoise

## Preprocessing of raw images -- Pixelwise denoising
## Image registration and motion correction
* Registration to a single fish with single modularity (motion correction)
* Registration to a single fish with multiple modularities from the same fish
* Registration of a single fish to a well-known brain atlas
* Average registration across fishes

## Cell segmentation
* Denoise (local PCA)
* Demix (local NMF, initialization using superpixels)

## Post-hoc analyses
### Spike and subthreshold activity extraction
* Spike detection -- neural network
* Subthreshold activity -- L1 Trend filter
### Cell selectivity
### Oscillations
### State space model
* Brain states and change detections

## Extended -- Behavioral data

## Extended -- Single-cell RNA seq
https://github.com/zqwei/single_cell_zebrafish
