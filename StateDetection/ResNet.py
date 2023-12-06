import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Define the layers of the Residual Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolutional layer.
            downsample (nn.Sequential): Downsample layer.

        Returns:
            None

        """

        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample

    def forward(self, x):
        """
        Define the forward pass of the Residual Block.

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


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes=5,
        input_size=(64, 64),
        channel_size=3,
        layers=[3, 4, 6, 3],
        args=None,
    ):
        super(ResNet, self).__init__()
        """
        Define the layers of the CNN model.

        Args:
            num_classes (int): Number of classes in the dataset.
            input_size (tuple): Size of the input image.
            args (argparse): Command line arguments.

        Returns:
            None

        """

        self.num_classes = num_classes
        self.input_size = input_size
        self.device = args.device
        print("Device: ", self.device)
        print("Input size: ", self.input_size)
        print("Num classes: ", self.num_classes)
        print("Channel size: ", channel_size)

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_size, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # [input_size x input_size x 3] -> [input_size x input_size x 64]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # [input_size x input_size x 64] -> [input_size/2 x input_size/2 x 64]
        self.layer1 = self.make_layer(ResidualBlock, 64, layers[0], stride=1) # [input_size/2 x input_size/2 x 64] -> [input_size/2 x input_size/2 x 64]
        self.layer2 = self.make_layer(ResidualBlock, 128, layers[1], stride=2) # [input_size/2 x input_size/2 x 64] -> [input_size/4 x input_size/4 x 128]
        self.layer3 = self.make_layer(ResidualBlock, 256, layers[2], stride=2) # [input_size/4 x input_size/4 x 128] -> [input_size/8 x input_size/8 x 256]
        self.layer4 = self.make_layer(ResidualBlock, 512, layers[3], stride=2) # [input_size/8 x input_size/8 x 256] -> [input_size/16 x input_size/16 x 512]
        # if self.input_size[0] = 32, then [input_size/16 x input_size/16 x 512] = [2 x 2 x 512], then we need to avgpool to [1 x 1 x 512] with kernel_size=2, stride=1
        # that is, if our input image is only 32x32, we can only  
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1) 
        self.fc = nn.Linear(512, num_classes)

        print(
            "Total number of parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )
        # print(self)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        print("out.shape", out.shape)
        print("[input_size/16 x input_size/16 x 512]", self.input_size[0]/16, self.input_size[1]/16, 512)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
