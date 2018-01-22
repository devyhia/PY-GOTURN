MODEL='/home/devyhia/PY-GOTURN/py_model_50000.pth'
DEPLOY_PROTO='./nets/tracker.prototxt'		 
DEPLOY_PROTO='./nets/tracker_1_fcs.prototxt'
#CAFFE_MODEL='./nets/models/pretrained_model/tracker.caffemodel'
CAFFE_MODEL='./dump_model_1_fcs_iter_50000.caffemodel'
TEST_DATA_PATH='/home/devyhia/VOT'		

python -m goturn.test.show_tracker_vot \
	--m $MODEL \
	--i $TEST_DATA_PATH 
