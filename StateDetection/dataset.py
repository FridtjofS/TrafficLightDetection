'''

import a test pytorch dataset

'''

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import random
import matplotlib.pyplot as plt

# load a predefined dataset
class StateDetectionDataset(Dataset):
    def __init__(self, data_dir, label_dir, input_size, num_classes, args):
        '''
        Load the dataset and perform preprocessing transformations.

        Args:
            data_dir (str): Path to the data directory.
            label_dir (str): Path to the label directory.
            input_size (tuple): Size of the input image.
            num_classes (int): Number of classes in the dataset.
            args (argparse): Command line arguments.

        Returns:
            None

        '''

        self.data_dir = data_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = args.device

        # Define transformations to be applied to the data
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load the data and label
        self.data, self.label = self.load_data()

    def __getitem__(self, index):
        '''
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            sample (tensor): [3, 64, 64] Sample image tensor
            label (tensor): [num_classes] Label tensor

        '''

        # Get sample image and label
        sample = self.data[index]
        label = self.label[index]

        # Apply transformations to the sample image
        sample = self.transform(sample)

        # Convert label to one-hot encoding
        label = torch.zeros(self.num_classes).scatter_(0, torch.tensor(label).unsqueeze(0), 1)

        return sample, label

    def __len__(self):
        '''
        Get the length of the dataset.

        Args:
            None

        Returns:
            length (int): Length of the dataset

        '''

        return len(self.data)

    def load_data(self):
        '''
        Load the data and label from the data and label directories.

        Args:
            None

        Returns:
            data (list): List of sample images
            label (list): List of sample labels

        '''

        # Load the data
        data = []
        for file in os.listdir(self.data_dir):
            # Load image
            image = plt.imread(os.path.join(self.data_dir, file))
            data.append(image)


