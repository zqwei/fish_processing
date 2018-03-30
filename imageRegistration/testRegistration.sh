dim=3
m=$2
f=$1
cnv=[1000x1000x100,1e-8,10]
# cnvSyN=[200x200x200x10,1e-7,10]
smth=3x2x1vox
down=4x2x1
# smthSyN=4x3x2x1x0vox
# downSyN=12x8x4x2x1
verb=1

echo out_${m%%.*}_

antsRegistration --dimensionality 3 --float 0 --verbose $verb \
                 --output [out_${m%%.*}_,out_${m%%.*}_Warped.nii.gz] \
                 --interpolation WelchWindowedSinc \
                 --use-histogram-matching 0 \
                 --initial-moving-transform [$f,$m,1] \
                 --transform Translation[0.1]\
                 --metric MI[$f,$m,1,32,Regular,0.25] \
                 --convergence $cnv \
                 --shrink-factors $down \
                 --smoothing-sigmas $smth
