DEPLOY_PROTO='./nets/tracker.prototxt'		 
#CAFFE_MODEL='./nets/models/pretrained_model/tracker.caffemodel'
CAFFE_MODEL='./dump_model_iter_100000.caffemodel'
TEST_DATA_PATH='/home/devyhia/VOT'		

python -m goturn.test.show_tracker_vot \
	--p $DEPLOY_PROTO \
	--m $CAFFE_MODEL \
	--i $TEST_DATA_PATH \
	--g 0
