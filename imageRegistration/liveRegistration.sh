# example code
# ./liveRegistration.sh ~/Downloads/zbb/stack_images/Transgenic/gfap-GFP.tif gfap-gcamps.tif

dim=3
m=$2
f=$1
cnv=[200x200x200x0,1e-8,10]
cnvSyN=[200x200x200x10,1e-7,10]
smth=4x3x2x1vox
down=12x8x4x2
smthSyN=4x3x2x1x0vox
downSyN=12x8x4x2x1
verb=1
interpolation=Linear #WelchWindowedSinc

antsRegistration --dimensionality 3 --float 1 --verbose $verb \
                 --output [output/${m%%.*}_,out_${m%%.*}_Warped.nii.gz] \
                 --interpolation $interpolation \
                 --use-histogram-matching 0 \
                 --initial-moving-transform [$f,$m,1] \
                 --transform Rigid[0.1] \
                 --metric MI[$f,$m,1,32,Regular,0.25] \
                 --convergence $cnv \
                 --shrink-factors $down \
                 --smoothing-sigmas $smth \
                 --transform Affine[0.1] \
                 --metric MI[$f,$m,1,32,Regular,0.25] \
                 --convergence $cnv \
                 --shrink-factors $down \
                 --smoothing-sigmas $smth
                 --transform SyN[0.05,6,0.5] \
                 --metric CC[$f,$m,1,2] \
                 --convergence $cnvSyN \
                 --shrink-factors $downSyN \
                 --smoothing-sigmas $smthSyN
