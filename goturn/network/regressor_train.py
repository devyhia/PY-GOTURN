# Date: Thursday 20 July 2017
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: 

import sys
import numpy as np
from regressor import regressor
from ..helper import config
import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable

from visdom import Visdom
viz = Visdom()

use_gpu = torch.cuda.is_available()
visualize_every = 10

base_lr = 0.000001
gamma = 0.1
stepsize = 100000
display = 1
# max_iter = 450000
momentum = 0.9
weight_decay = 0.0005
snapshot = 50000

class regressor_train:

    """Docstring for regressor_train. """

    def __init__(self, logger, pretrained_model=None):
        """TODO: to be defined1. """

        self.kDoTrain = True
        self.kNumInputs = 3
        self.kLearningRate = base_lr
        self.regressor = regressor(
            self.kNumInputs, logger, 
            train=self.kDoTrain, pretrained_model=pretrained_model)
        self.optimizer = optim.SGD(
            self.regressor.model.classifier.parameters(),
            lr=self.kLearningRate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        self.logger = logger
        self.current_step = 0
        

    def set_boxes_gt(self, bboxes_gt):
        """TODO: Docstring for set_boxes_gt.
        :returns: TODO
        """
        num_images = len(bboxes_gt)
        input_dims = 4

        self.regressor.bboxes = np.zeros((num_images, input_dims, 1, 1))

        for i in range(num_images):
            bbox_gt = bboxes_gt[i]
            bbox = np.asarray([bbox_gt.x1, bbox_gt.y1, bbox_gt.x2, bbox_gt.y2])
            bbox = bbox.reshape(bbox.shape[0], 1, 1)
            self.regressor.bboxes[i] = bbox
    

    def update_learning_rate(self):
        """
        Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
        """

        if self.current_step % stepsize == 0:
            self.kLearningRate *= gamma

            print('===> LR is set to {}'.format(self.kLearningRate))

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.kLearningRate


    def train(self, images, targets, bboxes_gt):
        """TODO: Docstring for train.
        :returns: TODO
        """

        logger = self.logger

        if self.regressor.train != True:
            logger.error('Phase is not to TRAIN!')


        if len(images) != len(targets):
            logger.error('Error = {} images but {} targets', len(images), len(targets))

        if len(images) != len(bboxes_gt):
            logger.error('Error = {} images but {} bboxes_gt', len(images), len(bboxes_gt))


        self.set_boxes_gt(bboxes_gt)
        self.regressor.set_images(images, targets)
        self.step()

    def visualize_train(self):
        viz.images(self.regressor.images, opts=dict(title='Random Images!', caption='How random.'))
        viz.images(self.regressor.targets, opts=dict(title='Random Targets!', caption='How random.'))

    def step(self):
        images, targets, bboxes = self.regressor.images, self.regressor.targets, self.regressor.bboxes

        self.logger.info('images: %d' % len(images))

        x1  = torch.from_numpy(images).float()
        x2  = torch.from_numpy(targets).float()
        y   = torch.from_numpy(bboxes).float()

        # wrap them in Variable
        if use_gpu:
            x1, x2, y = Variable(x1.cuda()), \
                Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
        else:
            x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward
        output = self.regressor.model(x1, x2)
        loss = self.regressor.loss_fn(output, y)

        # backward + optimize
        loss.backward()
        self.optimizer.step()

        # statistics
        if self.current_step % display == 0:
            self.logger.info('[training] step = %d, loss = %f' % (self.current_step, loss.data[0]))

        self.current_step += 1

        # Update Learning Rate
        self.update_learning_rate()

        # self.visualize_train()
