import random
import numpy as np
import torch
import os


def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    
    
    
    

    

def print2way(f, *x):
    print(*x)
    print(*x, file=f)
    f.flush()

def plot_loss_acc(val_losses, train_losses, val_accs, train_accs, dir):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()
    ax1.plot(val_losses, 'b', label='Validation loss')
    ax1.plot(train_losses, 'b', label='Train loss', linestyle='dashed', alpha=0.5)
    ax2.plot(val_accs, 'r', label='Validation accuracy')
    ax2.plot(train_accs, 'r', label='Train accuracy', linestyle='dashed', alpha=0.5)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.legend(loc='upper left')
    ax1.legend(loc='lower left')
    plt.title('Loss and accuracy')
    fig.tight_layout()
    plt.savefig(os.path.join(dir, 'loss_acc.svg'), format='svg')
    plt.close()
