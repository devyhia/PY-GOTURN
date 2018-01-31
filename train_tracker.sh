IMAGENET_FOLDER='/home/devyhia/ILSVRC2014'
ALOV_FOLDER='/home/devyhia/ALOV'
INIT_CAFFEMODEL='dump_model_4_fcs__iter_50000.caffemodel'
#INIT_CAFFEMODEL='nets//models/pretrained_model/tracker.caffemodel'
TRACKER_PROTO='nets/tracker_4_fcs.prototxt'
# TRACKER_PROTO='nets/tracker_ensemble1.prototxt'
SOLVER_PROTO='nets/solver_4_fcs.prototxt'
PROC='4FCS'

python -m goturn.train.train \
--proc $PROC \
--imagenet $IMAGENET_FOLDER \
--alov $ALOV_FOLDER \
--init_caffemodel $INIT_CAFFEMODEL \
--train_prototxt $TRACKER_PROTO \
--solver_prototxt $SOLVER_PROTO \
--lamda_shift 5 \
--lamda_scale 15 \
--min_scale -0.4 \
--max_scale 0.4 \
--gpu_id 0 