import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from time import strftime
import pickle
import random
import sys


from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets

#from torchsummary import summary

from model import StateDetection
#from StateDetection.ResNet_simple import ResNet
from ResNet_withBottleneck import ResNet
from dataset import StateDetectionDataset
# import train test split from torch
#from sklearn.model_selection import train_test_split
from utils import *


#import cv2


#class RandomAffineNearest(transforms.RandomAffine):
#    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=None):
#        super().__init__(degrees, translate, scale, shear, resample, fillcolor)
#
#    def __call__(self, img):
#        """
#        Args:
#            img (PIL Image): Image to be transformed.
#
#        Returns:
#            PIL Image: Affine transformed image.
#        """
#        angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
#        center = (img.size[0] / 2, img.size[1] / 2)
#        matrix = F._get_inverse_affine_matrix(center, angle, translations, scale, shear)
#        
#        # Convert PIL Image to OpenCV format (numpy array)
#        img_np = np.array(img)
#        
#        # Apply affine transformation using OpenCV
#        transformed_img = cv2.warpAffine(img_np, np.array(matrix).reshape(2, 3), dsize=img.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
#        
#        # Convert back to PIL Image
#        return Image.fromarray(transformed_img)
        

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

    cwd = os.getcwd()

    # Load dataset
    if args.data_dir == "SVHN":
        train_dataset = datasets.SVHN(root=os.path.join(cwd, "data"), split="train", download=True, transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        val_dataset = datasets.SVHN(root=os.path.join(cwd, "data"), split="test", download=True, transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        args.input_sizes = (train_dataset.data.shape[2], train_dataset.data.shape[3])
        args.channel_size = train_dataset.data.shape[1]
        args.num_classes = 10
        print2way(logf, "Train dataset shape: ", train_dataset.data.shape)
        print2way(logf, "Train dataset labels shape: ", train_dataset.labels.shape)
    elif args.data_dir == "MNIST":
        train_dataset = datasets.MNIST(root=os.path.join(cwd, "data"), train=True, download=True, transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        val_dataset = datasets.MNIST(root=os.path.join(cwd, "data"), train=False, download=True, transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        args.input_sizes = (train_dataset.data.shape[1], train_dataset.data.shape[2])
        args.channel_size = 1
        args.num_classes = 10
        print2way(logf, "Train dataset shape: ", train_dataset.data.shape)
        print2way(logf, "Train dataset labels shape: ", train_dataset.targets.shape)
    elif args.data_dir == "CIFAR10":
        transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.CIFAR10(root=os.path.join(cwd, "data"), train=True, download=True, transform=transforms_train)
        val_dataset = datasets.CIFAR10(root=os.path.join(cwd, "data"), train=False, download=True, transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]))
        args.input_sizes = (train_dataset.data.shape[1], train_dataset.data.shape[2])
        args.channel_size = train_dataset.data.shape[3]
        args.num_classes = 10
        print2way(logf, "Train dataset shape: ", train_dataset.data.shape) # (50000, 32, 32, 3)
        print2way(logf, "Train dataset labels shape: ", len(train_dataset.targets)) # 50000
    elif args.data_dir == "TrafficLight":
        #train_transform_old = transforms.Compose([
        #
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.4, hue=0),
        #    
        #    #transforms.Pad(64 , padding_mode="constant", fill= (255, 0, 0)), # for visualization
        #    transforms.Pad(64 , padding_mode="edge"), 
        #    transforms.RandomRotation(15),
        #    transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
        #    transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 0.5), ratio=(1, 1)), 
        #
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.1307,), (0.3081,)),
        #])

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.4, hue=0),
            transforms.RandomRotation(15),
            #transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
            transforms.RandomResizedCrop(size=args.input_sizes, scale=(0.5, 0.5), ratio=(1, 1)), 
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        val_transform_resize = transforms.Compose([
            transforms.Resize(args.input_sizes),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        val_transform_centercrop = transforms.Compose([
            transforms.CenterCrop(args.input_sizes),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        val_transform_randomresizedcrop = transforms.Compose([
            transforms.RandomResizedCrop(size=args.input_sizes, scale=(0.5, 0.5), ratio=(1, 1)), 
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = StateDetectionDataset(train=True, data_dir=os.path.join(cwd, args.train_data_dir), transform=train_transform, input_size=args.input_sizes, args=args)
        val_dataset = StateDetectionDataset(train=False, data_dir=os.path.join(cwd, args.train_data_dir), transform=val_transform_centercrop, input_size=args.input_sizes, args=args)
        
        args.channel_size = train_dataset.data.shape[3]
        args.num_classes = 5
        print2way(logf, "Train dataset shape: ", train_dataset.data.shape) # (40, 128, 128, 3)
        print2way(logf, "Val dataset shape: ", val_dataset.data.shape) # (40, 128, 128, 3)
        print2way(logf, "Train dataset labels shape: ", len(train_dataset.label)) # 40
        print2way(logf, "Val dataset labels shape: ", len(val_dataset.label))
        print2way(logf, "Train transform: ", train_transform)
        


    # if the batch size does not divide the dataset size, the last batch will be smaller
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    if args.predefined_model == "resnet10":
        args.resnet_layers = [1, 1, 1, 1]
        args.resnet_output_channels = [64, 128, 256, 512]
        args.resnet_block = "simple"
    elif args.predefined_model == "resnet18":
        args.resnet_layers = [2, 2, 2, 2]
        args.resnet_output_channels = [64, 128, 256, 512]
        args.resnet_block = "simple"
    elif args.predefined_model == "resnet34":
        args.resnet_layers = [3, 4, 6, 3]
        args.resnet_output_channels = [64, 128, 256, 512]
        args.resnet_block = "simple"
    elif args.predefined_model == "resnet50":
        args.resnet_layers = [3, 4, 6, 3]
        args.resnet_output_channels = [64, 128, 256, 512]
        args.resnet_block = "bottleneck"
    elif args.predefined_model == "resnet101":
        args.resnet_layers = [3, 4, 23, 3]
        args.resnet_output_channels = [64, 128, 256, 512]
        args.resnet_block = "bottleneck"
    elif args.predefined_model == "resnet152":
        args.resnet_layers = [3, 8, 36, 3]
        args.resnet_output_channels = [64, 128, 256, 512]
        args.resnet_block = "bottleneck"
    elif args.predefined_model == "resnet200":
        args.resnet_layers = [3, 24, 36, 3]
        args.resnet_output_channels = [64, 128, 256, 512]
        args.resnet_block = "bottleneck"

    

    model = ResNet(
        num_classes=args.num_classes,
        input_size=args.input_sizes,
        channel_size=args.channel_size,
        layers=args.resnet_layers,
        out_channels=args.resnet_output_channels,
        blocktype=args.resnet_block,
        logf=logf,
        args=args,
    )

    #model = StateDetection(
    #    num_classes=args.num_classes,
    #    input_size=args.input_sizes,
    #    channel_size=args.channel_size,
    #    logf=logf,
    #    args=args,
    #)

    #plot a batch of images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    import torchvision.utils as vutils
    img_grid = vutils.make_grid(images, normalize=True)
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.title("Sample Training images")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_model_dir, "sample_images.png"))
    plt.close()
    # same for validation set
    dataiter = iter(val_loader)
    images, labels = next(dataiter)
    img_grid = vutils.make_grid(images, normalize=True)
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.title("Sample Validation images")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_model_dir, "sample_val_images.png"))
    plt.close()
    

    #print2way(logf, summary(model, (args.channel_size, 224, 224)))
    # Define optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    if args.annealer == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0001)
    elif args.annealer == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.annealer == "linear":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        raise Exception("Invalid annealer")
    model.to(device)

    # Initialize best validation accuracy
    best_val_acc = 0
    val_loss_list = []
    train_loss_list = []
    val_acc_list = []
    train_acc_list = []

    print2way(logf, "\nArguments:\n")
    for arg in vars(args):
        print2way(logf, f"{arg}: {getattr(args, arg)}")
    print2way(logf, "\n\n")
    
    start_time = time.time()

    # Training loop
    for epoch in range(args.num_epochs):
        train_loss = 0
        train_acc = 0
        val_acc = 0
        val_loss = 0
        epoch_start_time = time.time()
        model.train()

        #lr_exp = torch.linspace(-3, 0, len(train_loader))
        #lrs = 10 ** lr_exp

        #lossi = []
        #lri = []

        # Loop over each batch from the training set
        for batch_idx, (data, target) in enumerate(train_loader):
            ###
            #lr = lrs[batch_idx]
            #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            ###
            data, target = data.to(device), target.to(device).long()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            #lossi.append(loss.item())
            #lri.append(lr)

            pred = torch.argmax(output, dim=1)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(data)
            train_acc += acc

            # Print training status
            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                print2way(logf, 
                    f"Train Epoch: {epoch} [{batch_idx * args.batch_size}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]",
                    f"\tLoss: {loss:.6f}\tAccuracy: {acc:.6f}"
                )

        #### plot learning rate and loss
        #plt.plot(lri, lossi)
        #plt.title("Learning rate and loss")
        #plt.xlabel("Learning rate")
        #plt.ylabel("Loss")
        ##plt.xscale("log")
        ##plt.yscale("log")
        #plt.tight_layout()
        #plt.savefig(os.path.join(args.save_model_dir, "lr_loss.png"))
        #plt.close()

        


        scheduler.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        model.eval()
        
        # Disable gradient calculation
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).long()
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
            f"\nEpoch: {epoch}\tTrain Loss: {train_loss:.6f}\tTrain Acc: {train_acc:.6f}\tVal Loss: {val_loss:.6f}",
            f"\tVal Acc: {val_acc:.6f}\tTime: {time.time() - epoch_start_time:.2f}s\n"
        )

        # Add loss to list
        val_loss_list.append(val_loss)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)


        # Save model if validation accuracy is greater than best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_model_dir, "model.pth"))
            args.load_model_dir = args.save_model_dir
            print2way(logf, "Model saved to %s" % args.save_model_dir)

        
        # Plot training loss and validation accuracy toegether in the same plot, but on different y axes
        plot_loss_acc(val_loss_list, train_loss_list, val_acc_list, train_acc_list, args.save_model_dir)
        
    # Save training loss and validation accuracy lists
    with open(os.path.join(args.save_model_dir, "train_loss_list.pkl"), "wb") as f:
        pickle.dump(train_loss_list, f)

    with open(os.path.join(args.save_model_dir, "val_acc_list.pkl"), "wb") as f:
        pickle.dump(val_acc_list, f)
    
    print2way(logf, "Total training time: ", strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    
    # plot final prediction on validation set
    test_val_acc = plot_final_prediction(args, val_loader, val_dataset, logf)

    return test_val_acc

def plot_final_prediction(args, val_loader, val_dataset, logf):
    """
    Plot final prediction on validation set

    Args:
        model (nn.Module): CNN model.
        val_loader (DataLoader): Validation data loader.
        save_model_dir (str): Directory to save model.
        num_classes (int): Number of classes.
        device (str): Device.

    Returns:
        None

    """
    # Set device
    #args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device

    if args.data_dir == "TrafficLight":
        args.num_classes = 5
    # load model
    model = ResNet(
        num_classes=args.num_classes,
        input_size=args.input_sizes,
        channel_size=args.channel_size,
        layers=args.resnet_layers,
        out_channels=args.resnet_output_channels,
        blocktype=args.resnet_block,
        logf=logf,
        args=args,

    )
    model.load_state_dict(torch.load(os.path.join(args.load_model_dir, "model.pth")))

    model.to(device)
    model.eval()

    # Load dataset
    #val_dataset = StateDetectionDataset(train=False, transform=transforms.Compose([
    #    #transforms.Resize((224, 224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,)),
    #]), args=args)
    
    label_names = val_dataset.label_names

    # Create data loader
    #val_loader = DataLoader(
    #    val_dataset,
    #    batch_size=args.batch_size,
    #    shuffle=False,
    #    num_workers=1,
    #)

    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    val_acc = 0
    correct = 0
    total_time = 0
    # Disable gradient calculation
    with torch.no_grad():
    
        # Loop over each batch from the validation set
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device).long()
            # measure time for inference (in milliseconds)
            epoch_start_time = time.time()
            output = model(data)
            end_time = time.time()
            
            print("end_time - epoch_start_time", end_time - epoch_start_time, "seconds")
            total_time += (end_time - epoch_start_time) * 1000 / len(data)
            pred = torch.argmax(output, dim=1)
            # also use the certainty of the prediction (softmax has not yet been applied inside the model)
            certainty = torch.max(F.softmax(output, dim=1), dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Update confusion matrix
            for i in range(len(target)):
                confusion_matrix[target[i]][pred[i]] += 1
            
            # only plot the last batch
            if  batch_idx == len(val_loader) - 1:
                val_acc = correct / len(val_dataset)
                print("last batch, size", len(data))
                # Plot final prediction
                fig, ax = plt.subplots(2,4)
                for i in range(8):
                    try:
                        sample_img = data[i].permute(1, 2, 0).cpu().numpy()
                        # unnormalize
                        sample_img *= 0.3081
                        sample_img += 0.1307
                        sample_img *= 255
                        sample_img = sample_img.astype(np.uint8)

                        ax[i//4][i%4].imshow(sample_img)
                        
                        pred_label = pred[i].item()
                        target_label = target[i].item()

                        pred_name = label_names[pred_label]
                        target_name = label_names[target_label]

                        color = "green" if pred_label == target_label else "black"
                        ax[i//4][i%4].set_title(f"Pred: {pred_name}\nActual: {target_name}\nCertainty: {certainty[0][i]*100:.0f}%", color=color)
                    except:
                        pass
                    ax.flat[i].axes.get_xaxis().set_visible(False)
                    ax.flat[i].axes.get_yaxis().set_visible(False)
                fig.suptitle(f"Validation Accuracy: {val_acc:.6f}")
                fig.tight_layout()
                plt.savefig(os.path.join(args.save_model_dir, "final_prediction.png"))
                plt.close()
            

    print2way(logf, "Average inference time: ", total_time / len(val_loader), "ms")

    # Plot confusion matrix
    # but first normalize the confusion matrix by row
    # but never divide by 0
    confusion_matrix_norm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis]  + 1e-6)

    
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix_norm)
    ax.set_xticks(np.arange(args.num_classes))
    ax.set_yticks(np.arange(args.num_classes))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            text = ax.text(j, i, f"{int(confusion_matrix[i, j])}",
                        ha="center", va="center", color="w")
    ax.set_title(f"Confusion Matrix for {len(val_dataset)} samples\nValidation Accuracy: {val_acc:.6f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    fig.tight_layout()
    plt.savefig(os.path.join(args.save_model_dir, "confusion_matrix.png"))
    plt.close()

    print2way(logf, "Final Validation Accuracy: ", val_acc)
    print2way(logf, "Final prediction saved to %s" % os.path.join(args.save_model_dir, "final_prediction.png"))

    return val_acc



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
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
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
    epoch_start_time = time.time()
    criterion = nn.CrossEntropyLoss()

    # Disable gradient calculation
    with torch.no_grad():
        # Loop over each batch from the test set
        for data, target in test_loader:
            # Move data and target to device
            data, target = data.to(device), target.to(device).long()
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
            test_loss, test_acc, time.time() - epoch_start_time
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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
    parser.add_argument("--save_model_dir", type=str, default=os.path.join(cwd, "models"), help="Directory to save model")
    parser.add_argument("--load_model_dir", type=str, default=os.path.join(cwd, "models"), help="Directory to load model")
    parser.add_argument("--data_dir", type=str, default="TrafficLight", help="Training data directory")
    parser.add_argument("--train_data_dir", type=str, default="augmented_dataset", help="train data directory")
    parser.add_argument("--resnet_layers", type=list, default=[1,1,1,1], help="Number of layers in each block")
    parser.add_argument("--resnet_output_channels", type=list, default=[64, 128, 256, 512], help="Number of output channels in each layer")
    parser.add_argument("--resnet_block", type=str, default="simple", help="Type of block")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--input_sizes", type=tuple, default=(64, 64), help="Input image size")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--channel_size", type=int, default=3, help="Number of channels")
    parser.add_argument("--predefined_model", type=str, default=None, help="Predefined model")
    parser.add_argument("--max_keep", type=int, default=1200, help="Maximum number of samples per class")
    parser.add_argument("--annealer", type=str, default="cosine", help="Learning rate annealer")

    args = parser.parse_args()

    # Set directory to save model
    custom_id = np.random.randint(0, 100000)

    # prepare grid search
    lrs = [0.00025]
    batch_sizes = [32]
    premodels = ["resnet10"]
    annealers = ["cosine"]

    best_params = {
        "lr": 0,
        "batch_size": 0,
        "premodel": "",
        "annealer": "",
    }

    total_combinations = len(lrs) * len(batch_sizes) * len(premodels) * len(annealers)
    i = 0

    save_model_dir = args.save_model_dir

    best_val_acc = 0
    
    
    for lr in lrs:
        for batch_size in batch_sizes:
            for annealer in annealers:
                for premodel in premodels:
                    i += 1
                    args.lr = lr
                    args.batch_size = batch_size
                    args.predefined_model = premodel
                    args.annealer = annealer

                    ### end of grid search    
                    ### Un-indent the following code to run normal training
                    ###################################################################
                    exp_dir = os.path.join(save_model_dir, f"model_{custom_id}")
                    if not os.path.exists(exp_dir):
                        os.makedirs(exp_dir)
                    args.save_model_dir = exp_dir
                    args.custom_id = custom_id

                    args.save_model_dir = os.path.join(args.save_model_dir, f"lr{lr}_batch{batch_size}_premodel{premodel}_annealer{annealer}")
                    os.makedirs(args.save_model_dir)
                    

                    # Set device
                    if args.device == "cuda" and not torch.cuda.is_available():
                        print("Warning: cuda not available, using cpu instead")
                        args.device = torch.device("cpu")
                    elif args.device == "cuda":
                        args.device = torch.device("cuda")
                    elif args.device == "dml":
                        try:
                            import torch_directml
                            args.device = torch_directml.device(torch_directml.default_device())
                            print("Using DirectML")
                        except:
                            pass
                    

                    print("args.save_model_dir", args.save_model_dir)

                    logf = open(os.path.join(args.save_model_dir, "log.txt"), "w")
                    args.logf = logf

                    val_acc = train(args, logf)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params["lr"] = lr
                        best_params["batch_size"] = batch_size
                        best_params["premodel"] = premodel
                        best_params["annealer"] = annealer


                    logf.close()

                    print("Best validation accuracy: ", best_val_acc)
                    print("-"*25)
                    print("Progress: ", i, "/", total_combinations)
                    print("-"*25)
                    

    print("Best validation accuracy: ", best_val_acc)
    
    with open(os.path.join(save_model_dir, f"model_{custom_id}", "best_params.txt"), "w") as f:
        print("\n\nBest validation accuracy: ", best_val_acc, file=f)
        for k, v in best_params.items():
            print(f"{k}: {v}", file=f)
            

    
    
    
   
    #if args.mode == "train":
    #    train(args, logf)
    #elif args.mode == "test":
    #    test(args, logf)
    #elif args.mode == "plot":
    #    plot_final_prediction(args, logf)
    #else:
    #    raise Exception("Invalid mode")

if __name__ == "__main__":
    main()


# good for overfitting:
# python train.py --predefined_model "resnet18" --data_dir "TrafficLight" --batch_size 16 --lr 0.005 --num_epochs 2  

# python train.py --device "dml" --max_keep 80 --num_epochs 3 --resnet_block "simple"