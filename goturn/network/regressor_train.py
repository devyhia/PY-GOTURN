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
from PIL import Image, ImageDraw

from visdom import Visdom
viz = Visdom()

use_gpu = torch.cuda.is_available()
visualize_every = 2

base_lr = 0.000001
gamma = 0.1
stepsize = 100000
display = 1
# max_iter = 450000
momentum = 0.9
weight_decay = 0.0005
snapshot = 1000

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
        
        trainable_weights = []
        trainable_bias = []

        for name, param in self.regressor.model.lstm.named_parameters():
            if 'weight' in name:
                trainable_weights.append(param)
            elif 'bias' in name:
                trainable_bias.append(param)
        
        for name, param in self.regressor.model.fc.named_parameters():
            if 'weight' in name:
                trainable_weights.append(param)
            elif 'bias' in name:
                trainable_bias.append(param)

        self.optimizer = optim.Adam(
            [
                {
                    'params': trainable_weights,
                    'lr': self.kLearningRate * 10
                },
                {
                    'params': trainable_bias,
                    'lr': self.kLearningRate * 20
                }
            ],
            lr=self.kLearningRate,
            # momentum=momentum,
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

    def visualize_train(self, y, output, loss):
        if self.current_step % visualize_every == 0:
            viz.images(self.regressor.images, opts=dict(title='Random Images!', caption='How random.'))
            viz.images(self.regressor.targets, opts=dict(title='Random Targets!', caption='How random.'))

            y = self.regressor.bboxes

            # Logging
            viz.text("""
                        Ground Truth: {} <br/>
                        Prediction: {} <br/>
                        Loss: {}
                    """.format(y[0], output[0], loss.data[0]),
                    win="Iteration__Text")

            # Visualization
            search_image = self.regressor.targets[0]
            search_image = search_image.reshape((self.regressor.height, self.regressor.width, self.regressor.channels))
            search_image += [104, 117, 123]
            search_image = search_image.astype(np.uint8)

            search_image = Image.fromarray(search_image)
            draw = ImageDraw.Draw(search_image)

            unscale_ratio = 10. / 227

            # (output.data.gpu() if use_gpu else output.data.cpu())
            # (y.data.gpu() if use_gpu else y.data.cpu())
            self.logger.info(y)
            self.logger.info(output)
            draw.rectangle(unscale_ratio * y[0], outline='red')
            draw.rectangle(unscale_ratio * output[0], outline='green')

            del draw

            viz.image(np.array(search_image).transpose(2, 0, 1), win="Iteration__Image", opts=dict(
                caption='Itr. {}'.format(self.current_step)
            ))

    def take_snapshot(self):
        """
        Saves a snapshot of the current weights
        """
        path = 'model_%s_%i.pth' % (self.regressor.model.name, self.current_step)
        if self.current_step % snapshot == 0:
            torch.save(self.regressor.model.state_dict(), path)

    def step(self):
        images, targets, bboxes = self.regressor.images, self.regressor.targets, self.regressor.bboxes

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

        # Take Snapshots
        self.take_snapshot()

        # Visualize Training
        # self.visualize_train(y, output, loss)
