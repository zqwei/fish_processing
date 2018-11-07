# Data processing
This is the data processing used for the calcium and voltage imaging

## Environment setup
* Optional packages:
    * [Caiman](https://github.com/flatironinstitute/CaImAn) -- code for visualization, basic statistics such as SNR, local correlation etc.
    * [FunImag](https://github.com/paninski-lab/funimag) -- superpixel cell segmentation
    * [Trefide](https://github.com/ikinsella/trefide) -- trend filtering, total variational denoise
    * [Fish](https://github.com/d-v-b/fish) -- basic analyses
    * Other packages required are shown https://github.com/zqwei/computer_setup

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

--------------------------
## Extended -- Behavioral data

## Extended -- Single-cell RNA seq
This has been merged as an untracked branch. Please contact in person for details.

--------------------------
## Depreciated old repos
If you are looking for / using one of the following repos, these are merged to independent folder in the current one, and they are no longer developed or supported.
* denoiseLocalPCA
* spike-detection-voltron
* cmos-denoise
* single-cell rna seq
--------------------------
## TODO
### Parallel processing
* Spark
* Parallel processing -- Dask
* GPU supports

### Merge git repos
* zfish_osc


### Sparse PCA
* Add prymid_tiles weight to fix squares in denoise movies

### Demix
* On-line version of NMF

### General problem
* long time series data
