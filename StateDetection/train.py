import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import datetime
import pickle
import random
import sys
import yaml

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets

from model import StateDetection
from dataset import StateDetectionDataset
from utils import *


def train(args):
    '''
    Train the CNN model.

    Args:
        args (argparse): Command line arguments.

    Returns:
        None

    '''

    # Set random seed for reproducibility
    set_seed(args.seed)

    # save the arguments to a yaml file
    with open(os.path.join(args.model_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args, f, default_flow_style=False)

    # Set device
    device = args.device

    #dataset = "MNIST" 
    dataset = "SVHN"

    # Load dataset
    #train_dataset = StateDetectionDataset(args.train_data_dir, args.train_label_dir, args.input_size, args.num_classes, args)
    #val_dataset = StateDetectionDataset(args.val_data_dir, args.val_label_dir, args.input_size, args.num_classes, args)
    # for now, load a predefined dataset from torch vision (SVHN), resize it to 64x64, and normalize it
    if dataset == "SVHN":
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]))
        val_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]))
        input_sizes = (train_dataset.data.shape[2], train_dataset.data.shape[3])
        channel_sizes = train_dataset.data.shape[1]
        num_classes = 10
        print("Train dataset shape: ", train_dataset.data.shape)
        print("Train dataset labels shape: ", train_dataset.labels.shape)
    elif dataset == "MNIST":
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]))
        val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]))
        input_sizes = (train_dataset.data.shape[1], train_dataset.data.shape[2])
        channel_sizes = 1
        num_classes = 10
        print("Train dataset shape: ", train_dataset.data.shape)
        print("Train dataset labels shape: ", train_dataset.targets.shape)


      

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    

    # plot the first 10 images in the training set
    #fig, axs = plt.subplots(2, 5)
    #fig.suptitle("First 10 images in the training set")
    #for i in range(2):
    #    for j in range(5):
    #        axs[i,j].imshow(train_dataset.data[i*5+j].transpose(1,2,0))
    #        axs[i,j].set_title(train_dataset.labels[i*5+j])
    #plt.show()

    model = StateDetection(num_classes= num_classes, 
                 input_size=input_sizes, 
                 channel_size=channel_sizes,
                 args=args)

    
                        
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # Initialize best validation accuracy
    best_val_acc = 0
    train_loss_list = []
    val_acc_list = []

    # Training loop
    for epoch in range(args.num_epochs):

        train_loss = 0
        train_acc = 0
        val_acc = 0
        val_loss = 0
        start_time = time.time()
        model.train()

        # Loop over each batch from the training set
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)
            loss.backward()
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Add loss to epoch
            train_loss += loss.item()

            # Calculate accuracy
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(data)
            train_acc += acc

            # Print training status
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), acc))
                
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        model.eval()

        # Disable gradient calculation
        with torch.no_grad():

            # Loop over each batch from the validation set
            for data, target in val_loader:
                
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                # Calculate accuracy
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(data)
                val_acc += acc

        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # Print training and validation results
        print('\nEpoch: {}\tTraining Loss: {:.6f}\tTraining Accuracy: {:.6f}\tValidation Loss: {:.6f}\tValidation Accuracy: {:.6f}\tTime: {:.2f}s\n'.format(
            epoch, train_loss, train_acc, val_loss, val_acc, time.time() - start_time))
        
        # Add loss to list
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)

        # Save model if validation accuracy is greater than best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pt'))
            print('Model saved to %s' % args.model_dir)

        # Plot training loss and validation accuracy toegether in the same plot, but on different y axes
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training loss', color=color)
        ax1.plot(train_loss_list, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Validation accuracy', color=color)
        ax2.plot(val_acc_list, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.savefig(os.path.join(args.model_dir, 'train_loss_val_acc.png'))
        plt.close()



    # Save training loss and validation accuracy lists
    with open(os.path.join(args.model_dir, 'train_loss_list.pkl'), 'wb') as f:
        pickle.dump(train_loss_list, f)

    with open(os.path.join(args.model_dir, 'val_acc_list.pkl'), 'wb') as f:
        pickle.dump(val_acc_list, f)

    

    
def test(args):
    '''
    Test the CNN model.

    Args:
        args (argparse): Command line arguments.

    Returns:
        None

    '''

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Set device
    device = args.device

    # Load dataset
    test_dataset = StateDetectionDataset(args.test_data_dir, args.test_label_dir, args.input_size, args.num_classes, args)

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = StateDetection(args.num_classes, args.input_size, args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt')))

    model.to(device)
    model.eval()

    # Initialize test accuracy
    test_acc = 0
    test_loss = 0
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()

    # Disable gradient calculation
    with torch.no_grad():

        # Loop over each batch from the test set
        for data, target in test_loader:

            # Move data and target to device
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            test_loss += loss.item()

            # Calculate accuracy
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(data)
            test_acc += acc

            # Update confusion matrix
            for i in range(len(target)):
                confusion_matrix[target[i]][pred[i]] += 1

    # Calculate average test loss
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    # Print test results
    print('\nTest Loss: {:.6f}\tTest Accuracy: {:.6f}\tTime: {:.2f}s\n'.format(
        test_loss, test_acc, time.time() - start_time))
    
    # Plot confusion matrix
    plt.figure()
    plt.imshow(confusion_matrix)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(range(args.num_classes))
    plt.yticks(range(args.num_classes))
    plt.savefig(os.path.join(args.model_dir, 'confusion_matrix.png'))

    
def main():
    '''
    Main function.

    Args:
        None

    Returns:
        None

    '''

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or test')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--log_interval', type=int, default=30, help='Logging interval')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default=device, help='Device')

    # Parse arguments
    args = parser.parse_args()

    custom_id = np.random.randint(0, 100000)

    # Set model directory
    exp_dir = os.path.join(args.model_dir, f'model_{custom_id}')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    args.model_dir = exp_dir


    # Print arguments
    print('\nArguments:')
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))

    # Train or test
    if args.mode == 'train':
        train(args)

    elif args.mode == 'test':
        test(args)

    else:
        raise Exception('Invalid mode')
    
if __name__ == '__main__':
    main()
