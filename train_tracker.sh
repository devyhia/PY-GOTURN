IMAGENET_FOLDER='/home/devyhia/ILSVRC2014'
ALOV_FOLDER='/home/devyhia/ALOV'
PRE_MODEL='model_best_loss.pth'

python -m goturn.train.train \
--alov $ALOV_FOLDER \
--imagenet $IMAGENET_FOLDER \
--pretrained_model $PRE_MODEL \
--lamda_shift 5 \
--lamda_scale 15 \
--min_scale -0.4 \
--max_scale 0.4
