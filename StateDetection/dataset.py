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


from utils import print2way

# load a predefined dataset
class StateDetectionDataset(Dataset):
    def __init__(self, train=True, input_size=(128,128), num_classes=5, data_dir=".\..\StateDetection\sd_train_data", transform=None, args=None):
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
        self.random_seed = args.seed
        self.data_dir = data_dir
        self.input_size = input_size
        self.num_classes = num_classes
        self.label_names = ['off', 'red', 'red_yellow', 'yellow', 'green']
        logf = args.logf

        # Define transformations to be applied to the data
        self.transform = transform

        # Load the data and label
        self.data, self.label = self.load_data()

        if train:
            print2way(logf, "\nOriginal Distribution: ")
            for i in range(self.num_classes):
                print2way(logf, self.label_names[i], self.label.count(i))
            print2way(logf, "\n ")

        max_keep = args.max_keep
        # Split the data into training and validation sets
        if max_keep is not None:
            # shorten or bootstrap the data per class, and split the data
            self.data, self.label, self.val_data, self.val_label = self.keep_max_samples(max_keep, train, logf, train_ratio=0.8)
        else:
            # split the data without changing the number of samples per class
            self.data, self.label, self.val_data, self.val_label = self.split_data(self.data, self.label, train_ratio=0.8)

        # Set the data and label for training or validation
        if train:
            self.data = self.data
            self.label = self.label
        else:
            self.data = self.val_data
            self.label = self.val_label

    def keep_max_samples(self, max_keep, train, logf, train_ratio=0.8):
        '''

        Args:
            max_keep (int): maximum number of samples to keep per class
            train (bool): whether to keep samples for training or validation
            logf (file): log file

        Returns:
            None
        '''
        # only keep min_num_samples number of samples per class to even out hte dataset
        # first, group the data and label by label
        data_grouped_by_label = [[] for i in range(self.num_classes)]
        for i in range(len(self.data)):
            data_grouped_by_label[self.label[i]].append(self.data[i])
        
    
        data_train = [[] for i in range(self.num_classes)]
        label_train = [[] for i in range(self.num_classes)]
        data_val = [[] for i in range(self.num_classes)]
        label_val = [[] for i in range(self.num_classes)]

        for i in range(self.num_classes):
            # split the data into train and validation sets
            data_train[i], label_train[i], data_val[i], label_val[i] = self.split_data(data_grouped_by_label[i], [i]*len(data_grouped_by_label[i]), train_ratio=train_ratio)
        
        if train:
            print2way(logf, "Distribution BEFORE keeping max_keep samples per class: ")
            print("\nTrain set: ") 
            for i in range(self.num_classes):
                print2way(logf, self.label_names[i], len(label_train[i]))
            print("\nValidation set: ")
            for i in range(self.num_classes):
                print2way(logf, self.label_names[i], len(label_val[i]))
            print2way(logf, "\n ")
        
        # bootstrap the data in the train set (and in the validation set if needed)
        max_keep_train = int(max_keep * train_ratio)
        max_keep_val = int(0.2 * max_keep)
        for i in range(self.num_classes):
            if len(data_train[i]) == 0 or len(data_train[i]) == max_keep_train:
                continue
            elif len(data_train[i]) < max_keep_train:
                # randomly choose samples with replacement
                data = random.choices(list(data_train[i]), k=max_keep_train)
                label = [i] * max_keep_train
            elif len(data_train[i]) > max_keep_train:
                # randomly choose max_keep samples
                data = random.sample(list(data_train[i]), max_keep_train)
                label = [i] * max_keep_train
            data_train[i] = np.array(data)
            label_train[i] = np.array(label)

            if len(data_val[i]) == 0 or len(data_val[i]) == max_keep_val:
                continue
            elif len(data_val[i]) < max_keep_val:
                # randomly choose samples with replacement
                data = random.choices(list(data_val[i]), k=max_keep_val)
                label = [i] * max_keep_val
            elif len(data_val[i]) > max_keep_val:
                # randomly choose max_keep samples
                data = random.sample(list(data_val[i]), max_keep_val)
                label = [i] * max_keep_val
            data_val[i] = np.array(data)
            label_val[i] = np.array(label)


        # convert list to numpy array
        data = np.concatenate(data_train, axis=0)
        label = np.concatenate(label_train, axis=0)
        val_data = np.concatenate(data_val, axis=0)
        val_label = np.concatenate(label_val, axis=0)

        # re-shuffle the data and label
        new_order = list(range(len(data)))
        random.seed(self.random_seed)
        random.shuffle(new_order)
        data = data[new_order]
        label = label[new_order]

        new_order = list(range(len(val_data)))
        random.seed(self.random_seed)
        random.shuffle(new_order)
        val_data = val_data[new_order]
        val_label = val_label[new_order]


        if train:
            print2way(logf, "Distribution AFTER keeping max_keep samples per class: ")
            print("\nTrain set: ") 
            for i in range(self.num_classes):
                print2way(logf, self.label_names[i], len(label_train[i]))
            print("\nValidation set: ")
            for i in range(self.num_classes):
                print2way(logf, self.label_names[i], len(label_val[i]))
            print2way(logf, "\n ")

        return data, label, val_data, val_label
    
    def bootstrap_data(self, data, label, max_keep, train, logf):
        '''
        Bootstrap the data to have max_keep 

        Args:   
            data (list): List of sample images.
            label (list): List of sample labels.
            max_keep (int): Maximum number of samples to keep 
            train (bool): Whether to keep samples for training or validation.
            logf (file): Log file.

        Returns:
            data (list): List of sample images.
            label (list): List of sample labels.

        '''

        if len(data) == 0:
            return data, label
        elif len(data) < max_keep:
            # randomly choose samples with replacement
            data = random.choices(list(data), k=max_keep)
            label = [label[0]] * max_keep
            return data, label
        elif len(data) == max_keep:
            return data, label
        elif len(data) > max_keep:
            # randomly choose max_keep samples
            data = random.sample(list(data), max_keep)
            label = [label[0]] * max_keep
            return data, label
        
    def bootstrap_data_old(self, data, label, max_keep, train, logf):

        self.data = []
        self.label = []
        # if a class has more than max_keep samples, randomly choose max_keep samples
        # if a class has less than max_keep samples, randomly choose samples with replacement
        # if a class has no samples, skip it
        for i in range(self.num_classes):
            if len(data_grouped_by_label[i]) > max_keep:
                # randomly choose max_keep samples
                self.data += random.sample(data_grouped_by_label[i], max_keep)
                self.label += [i] * max_keep
            elif len(data_grouped_by_label[i]) > 0:
                # randomly choose samples with replacement
                self.data += random.choices(data_grouped_by_label[i], k=max_keep)
                self.label += [i] * max_keep
            else:
                # skip this class
                continue
                

        if train:
            print2way(logf, "Distribution after keeping max_keep samples per class: ")
            for i in range(self.num_classes):
                print2way(logf, self.label_names[i], self.label.count(i))
            print2way(logf, "\n ")

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
        #train_data = np.array(train_data)
        #val_data = np.array(val_data)
        train_data_arr = np.zeros((len(train_data), self.input_size[0], self.input_size[1], 3))
        val_data_arr = np.zeros((len(val_data), self.input_size[0], self.input_size[1], 3))
        for i in range(len(train_data)):
            train_data_arr[i] = train_data[i]
        for i in range(len(val_data)):
            val_data_arr[i] = val_data[i]

        train_data = train_data_arr
        val_data = val_data_arr

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
                try:
                    label_dict = json.load(f)
                except:
                    print("error in loading json file: ", label_list[i] )
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