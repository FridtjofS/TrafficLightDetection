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
from PIL import Image
import json


# load a predefined dataset
class StateDetectionDataset(Dataset):
    def __init__(self, train=True, input_size=(128,128), num_classes=5, data_dir=".\..\Annotation\GUI\sd_traffic_lights", transform=None, args=None):
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
        self.random_seed = 0
        self.data_dir = data_dir
        self.input_size = input_size
        self.num_classes = num_classes
        self.label_names = ['off', 'red', 'red_yellow', 'yellow', 'green']

        # Define transformations to be applied to the data
        self.transform = transform

        # Load the data and label
        self.data, self.label = self.load_data()

        # Split the data into training and validation sets
        self.data, self.label, self.val_data, self.val_label = self.split_data(self.data, self.label, train_ratio=0.8)

        # Set the data and label for training or validation
        if train:
            self.data = self.data
            self.label = self.label
        else:
            self.data = self.val_data
            self.label = self.val_label

    def split_data(self, data, label, train_ratio):
        '''
        Split the data into training and validation sets.

        Args:
            data (list): List of sample images.
            label (list): List of sample labels.
            train_ratio (float): Ratio of training samples to total samples.

        Returns:
            train_data (list): List of training samples.
            train_label (list): List of training labels.
            val_data (list): List of validation samples.
            val_label (list): List of validation labels.

        '''

        # Get the number of samples
        num_samples = len(data)

        # Get the number of training samples
        num_train = int(train_ratio * num_samples)

        # Shuffle the data and label, but maintain their correspondence
        new_order = list(range(num_samples))
        random.seed(self.random_seed)
        random.shuffle(new_order)
        data = [data[i] for i in new_order]
        label = [label[i] for i in new_order]

        # Split the data and label
        train_data = data[:num_train]
        train_label = label[:num_train]
        val_data = data[num_train:]
        val_label = label[num_train:]

        # convert list to numpy array
        train_data = np.array(train_data)
        val_data = np.array(val_data)

        return train_data, train_label, val_data, val_label
    

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

        # Convert the data type of the numpy ndarray to uint8
        sample = (sample * 255).astype(np.uint8)

        # Convert the numpy ndarray to a PIL Image
        sample = Image.fromarray(sample)

        # Apply transformations to the sample image
        if self.transform:
            sample = self.transform(sample)

        # Convert label to tensor
        label = torch.tensor(label)

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

        # empty array to store the data and label of size
        data = []
        label = []
        # they have the same name, the data has .jpg and the label has .json
        data_list = os.listdir(self.data_dir)
        label_list = os.listdir(self.data_dir)
        # filter the extension
        data_list = [i for i in data_list if (i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg'))]
        label_list = [i for i in label_list if (i.endswith('.json'))]

        # load the data
        for i in range(len(data_list)):
            # load the data
            img = transforms.ToTensor()(Image.open(os.path.join(self.data_dir, data_list[i])))
            img = img.permute(1,2,0)
            
            # add the image to the list
            data.append(img)

            
            # load the label
            with open(os.path.join(self.data_dir, label_list[i]), 'r') as f:
                label_dict = json.load(f)
            lb = label_dict['state']
            # add the label to the list
            label.append(lb)

            
        
        

        #connvert list to numpy array
        #data = np.array(data)
        #label = np.array(label)
        #print("type of data: ", type(data))
        #print("type of label: ", type(label))
        #print("type of data[0]: ", type(data[0]))
        #print("type of label[0]: ", type(label[0]))
            
        return data, label
    

if __name__ == "__main__":
    # test the dataset
    import argparse
    parser = argparse.ArgumentParser(description='Test the dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for PyTorch computation')
    args = parser.parse_args()

    dataset = StateDetectionDataset(args=args, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
        ]))
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    
    
        
    print(len(dataset))
    print("Samples per class: ")
    for i in range(dataset.num_classes):
        print(dataset.label_names[i], dataset.label.count(i))


    sample, label = next(iter(dataloader))
    for i in range(4):
        plt.subplot(2,2,i+1)
        sample_i = sample[i].clone()
        
        plt.imshow(sample_i.permute(1,2,0))
        plt.title(dataset.label_names[torch.argmax(label[i])])
        plt.axis('off')
    plt.show()