# GoTURN Model -- from: pygoturn's repo

import torch
from torchvision import models
import torch.nn as nn

class GoNet(nn.Module):
    ''' Neural Network class
        Two stream model:
    ________
   |        | conv layers              Untrained Fully
   |Previous|------------------>|      Connected Layers
   | frame  |                   |    ___     ___     ___
   |________|                   |   |   |   |   |   |   |   fc4
               Pretrained       |   |   |   |   |   |   |    * (left)
               CaffeNet         |-->|fc1|-->|fc2|-->|fc3|--> * (top)
               Convolution      |   |   |   |   |   |   |    * (right)
               layers           |   |___|   |___|   |___|    * (bottom)
    ________                    |   (4096)  (4096)  (4096)  (4)
   |        |                   |
   | Current|------------------>|
   | frame  |
   |________|

    '''
    def __init__(self):
        super(GoNet, self).__init__()
        self.name = "alexnet"
        alex = models.alexnet(pretrained=True)
        for param in alex.parameters():
            param.requires_grad = False
        self.features = alex.features
        # self.classifier = nn.LSTM(
        #     input_size=256*6*6*2,
        #     hidden_size=4096,
        #     num_layers=2,
        #     dropout=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(256*6*6*2, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4)
        )
        
        self.weight_init()
    
    def weight_init(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(1)
                m.weight.data.normal_(0, 0.005)

    # feed forward through the neural net
    def forward(self, x, y):
        x1 = self.features(x)
        x1 = x1.view(x.size(0), 256*6*6)
        x2 = self.features(y)
        x2 = x2.view(x.size(0), 256*6*6)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x
