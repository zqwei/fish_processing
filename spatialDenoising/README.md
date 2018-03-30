# voltagedenoising

Approach: Video -> M1 -> NMF -> (M2 or M3) -> denoised results

M1: denoise via greedy sparse PCA (several constraints)
M2: denoise via Trend filtering
M3: Neural Network denoising

# Examples:
1. demo trend filter: demo to detrend and and estimate noise
2. Greedy demo: demo of M1
3. Four-greedy-denoisers-demo: demo of M1, applied in 4 windows (also contains parallel example).

## Comments:
1. Runs in python3
2. We assume the data is motion corrected and detrended by your method of choice.
3. Use four greedy denoisers (M1 applied in 4 grids) if M1 alone results in several block artifacts.
4. Ongoing: NMF step for VI and M3 code.

To do list:
(1) Combine three methods in one pipeline (python and matlab)
(2) Run the methods on a few datasets and carefully quantify the results. 
(3) Compare against some baseline methods, eg vanilla CNMF or PCA-ICA


## CoCaIm - Compression (block- SVD)
## TreFiDe - Trend Filtering Denoising
Dependencies:
1. [CaImAn](https://github.com/simonsfoundation/CaImAn) - follow their setup instructions
2. [CVXPY](https://cvxgrp.github.io/cvxpy/install/index.html) - follow their setup instructions, however on Ubuntu and Py3.6 I found that I first needed to use conda to install blas and lapack as described in the last comment [here](https://github.com/cvxgrp/cvxpy/issues/357). 
3. [CVXOPT] (http://cvxopt.org/). Test installation with:https://scaron.info/blog/linear-programming-in-python-with-cvxopt.html/
