#DEPLOY_PROTO='./nets/tracker.prototxt'		 
MODEL='/home/devyhia/PY-GOTURN/model_best_loss.pth'
TEST_DATA_PATH='/home/devyhia/VOT'

python -m goturn.test.show_tracker_vot \
	--m $MODEL \
	--i $TEST_DATA_PATH 
