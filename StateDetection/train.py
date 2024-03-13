import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import sys
from time import strftime
import pickle

from torch.utils.data import DataLoader
from torchvision import datasets

# fix relative imports
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, path)
from ResNet import ResNet
from dataset import StateDetectionDataset
from utils import *


def train(args, logf):
    """
    Train the CNN model.

    Args:
        args (argparse): Command line arguments.
        logf (file): Log file.

    Returns:
        None

    """

    # Set random seed for reproducibility
    set_seed(args.seed)
    device = args.device

    cwd = os.getcwd()

    # Load dataset
    if args.dataset_name == "SVHN":
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
    elif args.dataset_name == "TrafficLight":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.4, hue=0),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(size=args.input_sizes, scale=(0.5, 0.5), ratio=(1, 1)), 
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Normalize((0.2998,), (0.2021,)), # mean and std of the standard dataset
            #transforms.Normalize((0.3215,), (0.2159,)),  # mean and std of the plus_tubi dataset
        ])
        val_transform_centercrop = transforms.Compose([
            transforms.CenterCrop(args.input_sizes),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Normalize((0.2998,), (0.2021,)), # mean and std of the standard dataset
            #transforms.Normalize((0.3215,), (0.2159,)),  # mean and std of the plus_tubi dataset
        ])
        train_dataset = StateDetectionDataset(train=True, data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)[:-3]), args.train_data_dir), transform=train_transform, input_size=args.input_sizes, args=args)
        val_dataset = StateDetectionDataset(train=False, data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)[:-3]), args.train_data_dir), transform=val_transform_centercrop, input_size=args.input_sizes, args=args)
        
        args.channel_size = train_dataset.data.shape[3]
        args.num_classes = 5
        print2way(logf, "Train dataset shape: ", train_dataset.data.shape) # (40, 128, 128, 3)
        print2way(logf, "Val dataset shape: ", val_dataset.data.shape) # (40, 128, 128, 3)
        print2way(logf, "Train dataset labels shape: ", len(train_dataset.label)) # 40
        print2way(logf, "Val dataset labels shape: ", len(val_dataset.label))
        print2way(logf, "Train transform: ", train_transform)
        print2way(logf, "mean of train dataset: ", train_dataset.data.mean())  # 0.2998
        print2way(logf, "std of train dataset: ", train_dataset.data.std()) # 0.2021
        print2way(logf, "mean of val dataset: ", val_dataset.data.mean()) # 0.3193
        print2way(logf, "std of val dataset: ", val_dataset.data.std()) # 0.2039
        


    # if the batch size does not divide the dataset size, the last batch will be smaller
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    # choose predefined model
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
    model.to(device)



    # plot a batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    import torchvision.utils as vutils
    img_grid = vutils.make_grid(images, normalize=True)
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.title("Sample Training images")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_model_dir, "sample_images.png"))
    plt.close()
    # plot a batch of validation images
    dataiter = iter(val_loader)
    images, labels = next(dataiter)
    img_grid = vutils.make_grid(images, normalize=True)
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.title("Sample Validation images")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_model_dir, "sample_val_images.png"))
    plt.close()
    

    # Define optimizer
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

        # Loop over each batch from the training set
        for batch_idx, (data, target) in enumerate(train_loader):
    
            data, target = data.to(device), target.to(device).long()
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
            #train_acc += acc
            train_acc += correct 

            # Print training status
            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                print2way(logf, 
                    f"Train Epoch: {epoch} [{batch_idx * args.batch_size}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]",
                    f"\tLoss: {loss:.6f}\tAccuracy: {acc:.6f}"
                )


        scheduler.step()
        train_loss /= len(train_loader)
        #train_acc /= len(train_loader)
        train_acc /= len(train_loader.dataset)
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
                #val_acc += correct / len(data)
                val_acc += correct

       
        val_loss /= len(val_loader)
        #val_acc /= len(val_loader)
        val_acc /= len(val_loader.dataset)

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
        val_loader (DataLoader): Validation data loader.
        val_dataset (Dataset): Validation dataset.
        logf (file): Log file.
        
    Returns:
        val_acc (float): Validation accuracy.

    """
    # Set device
    device = args.device

    if args.dataset_name == "TrafficLight":
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
    
    label_names = val_dataset.label_names
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    val_acc = 0
    correct = 0
    total_time = 0

    # Disable gradient calculation
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device).long()

            epoch_start_time = time.time()
            output = model(data)
            end_time = time.time()
            
            #print("end_time - epoch_start_time", end_time - epoch_start_time, "seconds")
            total_time += (end_time - epoch_start_time)
            pred = torch.argmax(output, dim=1)
            certainty = torch.max(F.softmax(output, dim=1), dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Update confusion matrix
            for i in range(len(target)):
                confusion_matrix[target[i]][pred[i]] += 1
            
            # only plot the last batch
            if  batch_idx == len(val_loader) - 1:
                val_acc = correct / len(val_dataset)
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
                        # if last batch is smaller than 8
                        pass
                    ax.flat[i].axes.get_xaxis().set_visible(False)
                    ax.flat[i].axes.get_yaxis().set_visible(False)
                fig.suptitle(f"Validation Accuracy: {val_acc:.6f}")
                fig.tight_layout()
                plt.savefig(os.path.join(args.save_model_dir, "final_prediction.png"))
                plt.close()
            

    print2way(logf, "Average validation inference time: ", total_time / len(val_loader), "s")

    # Plot confusion matrix
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
    plt.savefig(os.path.join(args.save_model_dir, "confusion_matrix.svg"), format='svg')
    plt.close()

    print2way(logf, "Final Validation Accuracy: ", val_acc)
    print2way(logf, "Final prediction saved to %s" % os.path.join(args.save_model_dir, "final_prediction.png"))

    return val_acc

def main():
    """
    Main function.

    Args:
        None

    Returns:
        None

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
    parser.add_argument("--save_model_dir", type=str, default=os.path.join(os.path.abspath(__file__)[:-3], '..', "models"), help="Directory to save model")
    parser.add_argument("--load_model_dir", type=str, default=os.path.join(os.path.abspath(__file__)[:-3], '..', "models"), help="Directory to load model")
    parser.add_argument("--dataset_name", type=str, default="TrafficLight", help="Training data directory")
    parser.add_argument("--train_data_dir", type=str, default="augmented_dataset", help="train data directory")
    parser.add_argument("--resnet_layers", type=list, default=[1,1,1,1], help="Number of layers in each block")
    parser.add_argument("--resnet_output_channels", type=list, default=[64, 128, 256, 512], help="Number of output channels in each layer")
    parser.add_argument("--resnet_block", type=str, default="simple", help="Type of block")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log_interval", type=int, default=5000, help="Logging interval")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--input_sizes", type=tuple, default=(64, 64), help="Input image size")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--channel_size", type=int, default=3, help="Number of channels")
    parser.add_argument("--predefined_model", type=str, default="resnet10", help="Predefined model")
    parser.add_argument("--max_keep", type=int, default=1400, help="Maximum number of samples per class")
    parser.add_argument("--annealer", type=str, default="cosine", help="Learning rate annealer")
    args = parser.parse_args()

    # Set directory to save model
    custom_id = np.random.randint(0, 100000)

    # Create directory to save model
    exp_dir = os.path.join(args.save_model_dir, f"model_{custom_id}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    args.save_model_dir = exp_dir
    args.custom_id = custom_id


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
    
    logf = open(os.path.join(args.save_model_dir, "log.txt"), "w")
    args.logf = logf

    # Train model
    train(args, logf)


if __name__ == "__main__":
    main()
