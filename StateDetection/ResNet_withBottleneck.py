import torch
import torch.nn as nn
import torch.nn.functional as F

# to access from either training or the Viztool
try:
    from utils import print2way
except:
    from StateDetection.utils import print2way
'''
Create a ResNet model which can work with both simple and bottleneck blocks.

First, define the simple block as a separate class. Then, define the bottleneck block as a 
separate class. Finally, define the ResNet model using the two block classes.
'''

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, is_first_block=False):
        """
        Define the layers of the Simple Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolutional layer.
            is_first_block (bool): Whether the block is the first block of the layer.

        Returns:
            None

        """

        super(SimpleBlock, self).__init__()
        self.is_first_block = is_first_block
        self.expansion = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        
        self.downsample = None
        if self.is_first_block and stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """
        Define the forward pass of the Simple Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, is_first_block=False, ):
        """
        Define the layers of the Bottleneck Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolutional layer.
            is_first_block (bool): Whether the block is the first block of the layer.
            

        Returns:
            None

        """

        super(BottleneckBlock, self).__init__()
        self.is_first_block = is_first_block
        self.expansion = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.relu = nn.ReLU()
        
        self.downsample = None
        if self.is_first_block:# and stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        """
        Define the forward pass of the Bottleneck Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(
        self,
        num_classes=5,
        input_size=(64, 64),
        channel_size=3,
        layers=[3, 4, 6, 3],
        out_channels=[64, 128, 256, 512],
        blocktype='bottleneck',
        logf=None,
        args=None,
        device='cpu',
    ):
        """
        Define the layers of the ResNet model.

        Args:
            num_classes (int): Number of classes.
            input_size (tuple): Size of the input image.
            channel_size (int): Number of input channels.
            layers (list): List of number of blocks in each layer.
            out_channels (list): List of number of output channels in each layer.
            blocktype (str): Type of block to use.
            logf (function): Logging function.
            args (dict): Dictionary of arguments.
            
        Returns:
            None

        """

        super(ResNet, self).__init__()
        self.logf = logf
        self.args = args
        self.device = args.device if args else device
        in_channels = 64 
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_size, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
        ) # size: input_size/2 x input_size/2 x 64
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # size: input_size/4 x input_size/4 x 64
        if blocktype == 'bottleneck':
            self.block = BottleneckBlock
            self.expansion = 4
        else:
            self.block = SimpleBlock
            self.expansion = 1
        
        self.layer1 = self._make_layer(self.block, in_channels, out_channels[0], layers[0], stride=1) # size: input_size/4 x input_size/4 x 64
        self.layer2 = self._make_layer(self.block, out_channels[0] * self.expansion, out_channels[1], layers[1], stride=2) # size: input_size/8 x input_size/8 x 128
        self.layer3 = self._make_layer(self.block, out_channels[1] * self.expansion, out_channels[2], layers[2], stride=2) # size: input_size/16 x input_size/16 x 256
        self.layer4 = self._make_layer(self.block, out_channels[2] * self.expansion, out_channels[3], layers[3], stride=2) # size: input_size/32 x input_size/32 x 512  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # size: 1 x 1 x 512
        self.fc = nn.Linear(out_channels[3] * self.expansion, num_classes) # size: 1 x 1 x num_classes
        self.to(self.device)
        
        if logf:
            print2way(logf, "\nTotal number of parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        

    def _make_layer(self, block, in_channels, out_channels, layers, stride=1):
        """
        Define the layers of the ResNet model.

        Args:
            block (class): Type of block to use.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            layers (list): List of number of blocks in each layer.
            stride (int): Stride of the convolutional layer.

        Returns:
            None

        """

        layers_list = []
        layers_list.append(block(in_channels, out_channels, stride=stride, is_first_block=True))
        for i in range(1, layers):
            layers_list.append(block(out_channels * self.expansion, out_channels))
        return nn.Sequential(*layers_list)
    
    def forward(self, x):
        """
        Define the forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
    
