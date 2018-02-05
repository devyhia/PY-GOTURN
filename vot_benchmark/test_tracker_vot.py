#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 theodoruszq@gmail.com
#
# Distributed under terms of the MIT license.

"""
Integrate tracker to VOT benchmark.
"""

import sys
import os
CODE_PTH= "/home/devyhia/PYGT2"
MODEL_PTH = "/home/devyhia/PY-GOTURN"
sys.path.insert(1, CODE_PTH)   # goturn module path
# Some lib path, need specify absolute path
# os.environ["LD_LIBRARY"] = "/usr/local/cuda-8.0/lib64"
# os.environ["PATH"] = "/home/zq/anaconda2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
os.environ['GLOG_minloglevel'] = '2'

from goturn.tracker.vot_tracker import tracker
from goturn.logger.logger import setup_logger
from goturn.network.regressor import regressor
from goturn.helper.BoundingBox import BoundingBox
import setproctitle
import argparse
import vot
import cv2

logger = setup_logger(logfile=None)

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", require=True, help="Path to the prototxt.")
# ap.add_argument("-m", "--model", require=True, help="Path to the model.")
# ap.add_argument("-g", "--gpuID", default=0, help="gpu to use")
# args = vars(ap.parse_args())

def main(model):
    setproctitle.setproctitle('VBT_PY_%s' % (model))
    args = {
            "model": "%s/%s" % (MODEL_PTH, model)
    }
    do_train = False

    objRegressor = regressor(
        num_inputs=1, 
        logger=logger,
        train=do_train,
        pretrained_model=args["model"]
    )
    objTracker = tracker(False, logger)

    handle = vot.VOT('rectangle')
    selection = handle.region()

    # Process the first frame
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    logger.info(imagefile)

    init_left, init_top, init_w, init_h = selection[:]

    # Feed minx, miny, maxx, maxy
    bbox = BoundingBox(init_left, init_top, init_left+init_w, init_top+init_h)
    bbox.frame_num = 1

    sMatImage = cv2.imread(imagefile)
    objTracker.init(sMatImage, bbox, objRegressor)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        sMatImage = cv2.imread(imagefile)
        est_bbox = objTracker.track(sMatImage)
        #logger.info("est_bbox: ", est_bbox)
        minx, miny = est_bbox.x1, est_bbox.y1
        maxx, maxy = est_bbox.x2, est_bbox.y2
        w, h = maxx - minx, maxy - miny
        selection = vot.Rectangle(minx, miny, w, h)
        #logger.info(selection)
        handle.report(selection)


