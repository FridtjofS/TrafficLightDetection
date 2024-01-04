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


from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets

from model import StateDetection
from ResNet import ResNet
from dataset import StateDetectionDataset
from utils import *


def train(args, logf):
    """
    Train the CNN model.

    Args:
        args (argparse): Command line arguments.

    Returns:
        None

    """

    # Set random seed for reproducibility
    set_seed(args.seed)
    device = args.device

    # Load dataset
    if args.data_dir == "SVHN":
        train_dataset = datasets.SVHN(root="./data", split="train", download=True, transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        val_dataset = datasets.SVHN(root="./data", split="test", download=True, transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        args.input_sizes = (train_dataset.data.shape[2], train_dataset.data.shape[3])
        args.channel_sizes = train_dataset.data.shape[1]
        args.num_classes = 10
        print2way(logf, "Train dataset shape: ", train_dataset.data.shape)
        print2way(logf, "Train dataset labels shape: ", train_dataset.labels.shape)
    elif args.data_dir == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        args.input_sizes = (train_dataset.data.shape[1], train_dataset.data.shape[2])
        args.channel_sizes = 1
        args.num_classes = 10
        print2way(logf, "Train dataset shape: ", train_dataset.data.shape)
        print2way(logf, "Train dataset labels shape: ", train_dataset.targets.shape)
    elif args.data_dir == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        args.input_sizes = (train_dataset.data.shape[1], train_dataset.data.shape[2])
        args.channel_sizes = train_dataset.data.shape[3]
        args.num_classes = 10
        print2way(logf, "Train dataset shape: ", train_dataset.data.shape)
        print2way(logf, "Train dataset labels shape: ", len(train_dataset.targets))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ResNet(
        num_classes=args.num_classes,
        input_size=args.input_sizes,
        channel_size=args.channel_sizes,
        layers=args.resnet_layers,
        logf=logf,
        args=args,
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    model.to(device)

    # Initialize best validation accuracy
    best_val_acc = 0
    train_loss_list = []
    val_acc_list = []

    print2way(logf, "\nArguments:\n")
    for arg in vars(args):
        print2way(logf, f"{arg}: {getattr(args, arg)}")
    print2way(logf, "\n\n")

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
            loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            pred = torch.argmax(output, dim=1)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(data)
            train_acc += acc

            # Print training status
            if batch_idx % args.log_interval == 0:
                print2way(logf, f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss:.6f}\tAccuracy: {acc:.6f}")
        
        scheduler.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        model.eval()

        # Disable gradient calculation
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                pred = torch.argmax(output, dim=1)
                correct = pred.eq(target.view_as(pred)).sum().item()
                val_acc += correct / len(data)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # Print training and validation results
        print2way(logf, 
            f"\nEpoch: {epoch}\tTraining Loss: {train_loss:.6f}\tTraining Accuracy: {train_acc:.6f}\
                \tValidation Loss: {val_loss:.6f}\tValidation Accuracy: {val_acc:.6f}\tTime: {time.time() - start_time:.2f}s\n"
        )

        # Add loss to list
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)

        # Save model if validation accuracy is greater than best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_model_dir, "model.pt"))
            print2way(logf, "Model saved to %s" % args.save_model_dir)

        # Plot training loss and validation accuracy toegether in the same plot, but on different y axes
        plot_loss_acc(train_loss_list, val_acc_list)

    # Save training loss and validation accuracy lists
    with open(os.path.join(args.save_model_dir, "train_loss_list.pkl"), "wb") as f:
        pickle.dump(train_loss_list, f)

    with open(os.path.join(args.save_model_dir, "val_acc_list.pkl"), "wb") as f:
        pickle.dump(val_acc_list, f)


def test(args, logf):
    """
    Test the CNN model.

    Args:
        args (argparse): Command line arguments.

    Returns:
        None

    """

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Set device
    device = args.device

    # Load dataset
    test_dataset = StateDetectionDataset(
        args.test_data_dir, args.test_label_dir, args.input_size, args.num_classes, args
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = StateDetection(args.num_classes, args.input_size, args)
    model.load_state_dict(torch.load(os.path.join(args.save_model_dir, "model.pt")))

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
    print2way(logf, 
        "\nTest Loss: {:.6f}\tTest Accuracy: {:.6f}\tTime: {:.2f}s\n".format(
            test_loss, test_acc, time.time() - start_time
        )
    )

    # Plot confusion matrix
    plt.figure()
    plt.imshow(confusion_matrix)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(args.num_classes))
    plt.yticks(range(args.num_classes))
    plt.savefig(os.path.join(args.save_model_dir, "confusion_matrix.png"))


def main():
    """
    Main function.

    Args:
        None

    Returns:
        None

    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
    parser.add_argument("--save_model_dir", type=str, default="models", help="Directory to save model")
    parser.add_argument("--data_dir", type=str, default="SVHN", help="Training data directory")
    parser.add_argument("--resnet_layers", type=list, default=[1, 1, 1, 1], help="Number of layers in each block")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--log_interval", type=int, default=30, help="Logging interval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--input_sizes", type=tuple, default=(128, 128), help="Input image size")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--channel_size", type=int, default=3, help="Number of channels")

    args = parser.parse_args()

    # Set directory to save model
    custom_id = np.random.randint(0, 100000)
    exp_dir = os.path.join(args.save_model_dir, f"model_{custom_id}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    args.save_model_dir = exp_dir
    args.custom_id = custom_id

    logf = open(os.path.join(args.save_model_dir, "log.txt"), "w")
    

    
   
    if args.mode == "train":
        train(args, logf)
    elif args.mode == "test":
        test(args, logf)
    else:
        raise Exception("Invalid mode")

if __name__ == "__main__":
    main()
