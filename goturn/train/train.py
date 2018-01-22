# Date: Friday 21 July 2017
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Training the tracker


import argparse
import setproctitle
from ..logger.logger import setup_logger
from ..loader.loader_imagenet import loader_imagenet
from ..loader.loader_alov import loader_alov
from ..train.example_generator import example_generator
from ..network.regressor_train import regressor_train
from ..tracker.tracker_trainer import tracker_trainer
import sys
import os
import numpy as np
import torch

logger = setup_logger(logfile=None)

ap = argparse.ArgumentParser()
ap.add_argument("-proc", "--proc", required = True, help = "Process Name")
ap.add_argument("-imagenet", "--imagenet", required = True, help = "Path to ImageNet folder")
ap.add_argument("-alov", "--alov", required = True, help = "Path to Alov folder")
ap.add_argument("-pretrained_model", "--pretrained_model", default=None, required = False, help = "Path to pytorch pretraianed model")
ap.add_argument("-lamda_shift", "--lamda_shift", required = True, help = "lamda shift")
ap.add_argument("-lamda_scale", "--lamda_scale", required = True, help = "lamda scale ")
ap.add_argument("-min_scale", "--min_scale", required = True, help = "min scale")
ap.add_argument("-max_scale", "--max_scale", required = True, help = "max scale")


RANDOM_SEED = 800
kNumBatches = 500000

def train_image(image_loader, images, tracker_trainer):
    """TODO: Docstring for train_image.
    """
    curr_image = np.random.randint(0, len(images))
    list_annotations = images[curr_image]
    curr_ann = np.random.randint(0, len(list_annotations))

    image, bbox = image_loader.load_annotation(curr_image, curr_ann)
    tracker_trainer.train(image, image, bbox, bbox)

def train_video(videos, tracker_trainer):
    """TODO: Docstring for train_video.
    """
    video_num = np.random.randint(0, len(videos))
    video = videos[video_num]

    annotations = video.annotations

    if len(annotations) < 2:
        logger.info('Error - video {} has only {} annotations', video.video_path, len(annotations))

    ann_index = np.random.randint(0, len(annotations) - 1)
    frame_num_prev, image_prev, bbox_prev = video.load_annotation(ann_index)

    frame_num_curr, image_curr, bbox_curr = video.load_annotation(ann_index + 1)
    tracker_trainer.train(image_prev, image_curr, bbox_prev, bbox_curr)

def main(args):
    """TODO: Docstring for main.
    """
    # Fix random seeds (numpy and pytorch) for reproducibility
    logger.info('Initializing pytorch..')
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    logger.info('Loading training data')
    # Load imagenet training images and annotations
    # imagenet_folder = os.path.join(args['imagenet'], 'images')
    # imagenet_annotations_folder = os.path.join(args['imagenet'], 'gt')
    # objLoaderImgNet = loader_imagenet(imagenet_folder, imagenet_annotations_folder, logger)
    # train_imagenet_images = objLoaderImgNet.loaderImageNetDet()

    # Load alov training images and annotations
    alov_folder = os.path.join(args['alov'], 'images')
    alov_annotations_folder = os.path.join(args['alov'], 'gt')
    objLoaderAlov = loader_alov(alov_folder, alov_annotations_folder, logger)
    objLoaderAlov.loaderAlov()
    train_alov_videos = objLoaderAlov.get_videos()[:1]

    # create example generator and setup the network
    objExampleGen = example_generator(float(args['lamda_shift']), float(args['lamda_scale']), float(args['min_scale']), float(args['max_scale']), logger)
    objRegTrain = regressor_train(logger, pretrained_model=args.get('pretrained_model', None))
    objTrackTrainer = tracker_trainer(objExampleGen, objRegTrain, logger)

    while objTrackTrainer.num_batches_ < kNumBatches:
        # train_image(objLoaderImgNet, train_imagenet_images, objTrackTrainer)
        train_video(train_alov_videos, objTrackTrainer)


if __name__ == '__main__':
    args = vars(ap.parse_args())
    setproctitle.setproctitle('TRAIN_TRACKER_%s' % args['proc'])
    main(args)
