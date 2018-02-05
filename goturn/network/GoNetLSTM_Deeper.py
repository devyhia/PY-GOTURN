# GoTURN Model -- from: pygoturn's repo

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.init as weight_init
from torch.autograd import Variable

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
    def __init__(self, init_weights=True):
        super(GoNet, self).__init__()
        self.name = "lstm_deeper"
        alex = models.alexnet(pretrained=True)
        for param in alex.parameters():
            param.requires_grad = False
        self.features = alex.features

        features_size = 256*6*6
        self.hidden_size = 1024
        self.num_classes = 4
        self.num_layers = 1
        self.time_steps = 2

        self.lstm = nn.LSTM(features_size, self.hidden_size, self.num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(self.hidden_size, self.num_classes)
        )

        if init_weights:
            self.weight_init()
    
    def weight_init(self):
        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    weight_init.normal(self.lstm.__getattr__(p), 0, 0.005)

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

        # Set initial states 
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        x1_aug = x1.unsqueeze(1)
        x2_aug = x2.unsqueeze(1)

        sequence = torch.cat((x1_aug, x2_aug), 1)

        # Forward propagate RNN
        out, _ = self.lstm(sequence, (h0, c0))

        # Decode hidden state of last time step
        x = self.classifier(out[:, -1, :])

        return x