# Date: Friday 02 June 2017 05:04:00 PM IST 
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Basic regressor function implemented

from __future__ import print_function
import os
import glob
import numpy as np
import sys
import cv2
from ..helper import config

# Torch Imports
import torch
import torchvision
from torch.autograd import Variable
from .GoNet import GoNet

use_gpu = torch.cuda.is_available()

if use_gpu:
    print('==> GPU is available :)')

class regressor:
    """Regressor Class"""

    def __init__(self, num_inputs, logger, train=True, pretrained_model=None):
        """TODO: to be defined"""

        self.num_inputs = num_inputs
        self.logger = logger
        self.pretrained_model = pretrained_model
        self.modified_params_ = False
        self.mean = [104, 117, 123]
        self.modified_params = False
        self.train = train
        
        self.model = GoNet()
        self.loss_fn = torch.nn.L1Loss()
        
        if pretrained_model:
            logger.info("=> loading checkpoint '{}'".format(pretrained_model))
            checkpoint = torch.load(pretrained_model)
            self.model.load_state_dict(checkpoint)
        
        if use_gpu:
            self.model = self.model.cuda()
            loss_fn = self.loss_fn.cuda()

        if train == True:
            logger.info('===> Phase: Train')
            self.model.train()
        else:
            logger.info('===> Phase: Test')
            self.model.eval()

        self.num_inputs = 1
        self.channels = 3
        self.height = 227
        self.width = 227

        # We are trying to change a Caffe model to PyTorch model.
        # We define self.targets, self.images and self.bbox as if they were blobs.
        self.images, self.targets, self.bboxes = None, None, None


    def set_images(self, images, targets):
        """TODO: Docstring for set_images.
        :returns: TODO
        """
        num_images = len(images)
        
        self.images = np.zeros((num_images, self.channels, self.height, self.width))
        self.targets = np.zeros((num_images, self.channels, self.height, self.width))
        for i in range(num_images):
            image = images[i]
            image_out = self.preprocess(image)
            self.images[i] = image_out

            target = targets[i]
            target_out = self.preprocess(target)
            self.targets[i] = target_out


    def preprocess(self, image):
        """TODO: Docstring for preprocess.

        :arg1: TODO
        :returns: TODO

        """
        num_channels = self.channels
        if num_channels == 1 and image.shape[2] == 3:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif num_channels == 1 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif num_channels == 3 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif num_channels == 3 and image.shape[2] == 1:
            image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_out = image

        if image_out.shape != (self.height, self.width, self.channels):
            image_out = cv2.resize(image_out, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        image_out = np.float32(image_out)
        image_out -= np.array(self.mean)
        image_out = np.transpose(image_out, [2, 0, 1])

        return image_out

    def regress(self, curr_search_region, target_region):
        return self.estimate(curr_search_region, target_region)

    def estimate(self, curr_search_region, target_region):
        curr_search_region = self.preprocess(curr_search_region)
        target_region = self.preprocess(target_region)

        self.images = curr_search_region.reshape(1, self.channels, self.height, self.width)
        self.targets = target_region(1, self.channels, self.height, self.width)

        x1  = torch.from_numpy(self.images).float()
        x2  = torch.from_numpy(self.targets).float()

        # wrap them in Variable
        if use_gpu:
            x1  = Variable(x1.cuda())
            x2  = Variable(x2.cuda())
        else:
            x1  = Variable(x1)
            x2  = Variable(x2)

        # forward
        output = self.model(x1, x2)
        bbox_estimate = output.data.numpy()

        return bbox_estimate
