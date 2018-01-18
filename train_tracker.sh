IMAGENET_FOLDER='/home/devyhia/ILSVRC2014'
ALOV_FOLDER='/home/devyhia/ALOV'
# PRE_MODEL='model_best_loss.pth'
INIT_CAFFEMODEL='nets/models/weights_init/tracker_init.caffemodel'
#INIT_CAFFEMODEL='nets//models/pretrained_model/tracker.caffemodel'
TRACKER_PROTO='nets/tracker_skip.prototxt'
# TRACKER_PROTO='nets/tracker_ensemble1.prototxt'
SOLVER_PROTO='nets/solver_skip.prototxt'
PROC='SKIP'

# --pretrained_model $PRE_MODEL \

python -m goturn.train.train \
--proc $PROC \
--imagenet $IMAGENET_FOLDER \
--alov $ALOV_FOLDER \
--imagenet $IMAGENET_FOLDER \
--lamda_shift 5 \
--lamda_scale 15 \
--min_scale -0.4 \
--max_scale 0.4