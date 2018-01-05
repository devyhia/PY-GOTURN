IMAGENET_FOLDER='~/ILSVRC2014'
ALOV_FOLDER='~/ALOV'

python -m goturn.train.train \
--imagenet $IMAGENET_FOLDER \
--alov $ALOV_FOLDER \
--lamda_shift 5 \
--lamda_scale 15 \
--min_scale -0.4 \
--max_scale 0.4