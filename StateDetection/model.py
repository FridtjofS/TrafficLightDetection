"""
Create a Convolutional Neural Network (CNN) model for the State Detection of a traffic light.

The images are cropped to only include the traffic light and resized to 64x64 pixels.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class StateDetection(nn.Module):
    def __init__(self, num_classes=5, 
                 input_size=(64, 64), 
                 channel_size=3,

                 args=None):
        super(StateDetection, self).__init__()
        '''
        Define the layers of the CNN model.

        Args:
            num_classes (int): Number of classes in the dataset.
            input_size (tuple): Size of the input image.
            args (argparse): Command line arguments.

        Returns:
            None

        '''

        self.num_classes = num_classes
        self.input_size = input_size
        self.device = args.device
        print("Device: ", self.device)
        print("Input size: ", self.input_size)
        print("Num classes: ", self.num_classes)


        self.net = nn.Sequential(
            nn.Conv2d(channel_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(2), # [input_size, input_size, 3] -> [input_size/2, input_size/2, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(2), # [input_size/2, input_size/2, 32] -> [input_size/4, input_size/4, 64]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(2), # [input_size/4, input_size/4, 64] -> [input_size/8, input_size/8, 128]
            nn.Flatten(),
            nn.Linear(128 * int(np.ceil(input_size[0]/8) * np.ceil(input_size[1]/8)), 128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        ).to(self.device)

        print("Total number of parameters: ", sum(p.numel() for p in self.net.parameters() if p.requires_grad))
    def forward(self, x):
        '''
        Forward pass of the CNN model.
        
        Args:
            x (tensor): [batch_size, 3, 64, 64] Input image tensor

        Returns:
            output (tensor): [batch_size, num_classes] Output tensor of probabilities for each class

        '''
        return self.net(x)

    def predict(self, x):
        '''
        Predict the class of the input image.
        
        Args:
            x (tensor): [batch_size, 3, 64, 64] Input image tensor
            
        Returns:
            output (tensor): [batch_size, 1] Output tensor of predicted classes
        '''
        with torch.no_grad():
            output = self.forward(x)
            output = torch.argmax(output, dim=1)

        return output
        

    def predict_prob(self, x):
        '''
        Predict the probability of each class of the input image.
        
        Args:
            x (tensor): [batch_size, 3, 64, 64] Input image tensor
            
        Returns:
            output (tensor): [batch_size, num_classes] Output tensor of probabilities for each class
        '''
        with torch.no_grad():
            output = self.forward(x)

        return output