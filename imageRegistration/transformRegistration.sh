dim=3
m=$2
f=$1
t=$3

antsApplyTransforms --dimensionality 3 -verbose 0 --float 1 \
                   --interpolation WelchWindowedSinc \
                   --input $m \
                   --reference-image $f \
                   --output out_${m%%.*}_Warped.nii.gz \
                   --transform out_${t%%.*}_Warped.nii.gz \
                   --transform out_${t%%.*}GenericAffine.mat \
