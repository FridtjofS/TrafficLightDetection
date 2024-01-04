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
    

def print2way(f, *x):
    print(*x)
    print(*x, file=f)
    f.flush()

def plot_loss_acc(train_losses, val_accs, dir):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(train_losses, 'b-')
    ax2.plot(val_accs, 'r-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Train loss', color='b')
    ax2.set_ylabel('Validation accuracy', color='r')
    plt.title('Loss and accuracy')
    fig.tight_layout()
    plt.savefig(os.path.join(dir, 'loss_acc.png'))
    plt.close()
