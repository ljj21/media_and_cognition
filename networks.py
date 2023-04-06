# ========================================================
#             Media and Cognition
#             Homework 2 Convolutional Neural Network
#             networks.py - Network definition
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2023
# ========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, in_channels, num_class, use_batch_norm=False, use_stn=False, dropout_prob=0):
        """
        Convolutional Neural Networks
        ----------------------
        :param in_channels: channel number of input image
        :param num_class: number of classes for the classification task
        :param use_batch_norm: whether to use batch normalization in convolutional layers and linear layers
        :param use_stn: whether to use spatial transformer network
        :param dropout_prob: dropout ratio of dropout layer which ranges from 0 to 1
        """
        super().__init__()

        if use_batch_norm:
            bn1d = nn.BatchNorm1d
            bn2d = nn.BatchNorm2d
        else:
            # use identity function to replace batch normalization
            bn1d = nn.Identity
            bn2d = nn.Identity

        if use_stn:
            self.stn = STN(in_channels)
        else:
            # use identity function to replace spatial transformer network
            self.stn = nn.Identity(in_channels)

        # >>> TODO 2.1: complete a multilayer convolutional neural network with nn.Sequential function.
        # input image with size [batch_size, in_channels, img_h, img_w]
        # Network structure:
        #        kernel_size  stride  padding  out_channels
        # conv       5          1        2          32
        # batchnorm
        # relu

        # conv       5          2        2          64
        # batchnorm
        # relu

        # maxpool    2          2        0

        # conv       3          1        1          64
        # batchnorm
        # relu

        # conv       3          1        1          128
        # batchnorm
        # relu

        # maxpool    2          2        0

        # conv       3          1        1          128
        # batchnorm
        # relu
        # dropout(p), where p is input parameter of dropout ratio
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=1, padding=2), bn2d(32), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2), bn2d(64), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), bn2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1), bn2d(128), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1), bn2d(
            128), nn.ReLU(), nn.Dropout(dropout_prob))
        # <<< TODO 2.1

        # >>> TODO 2.2: complete a sub-network with two linear layers by using nn.Sequential function
        # Hint: note that the size of input images is (3, 32, 32) by default, what is the size of
        # the output of the convolution layers?
        # Network structure:
        #          out_channels
        # linear       256
        # activation
        # batchnorm
        # dropout(p), where p is input parameter of dropout ratio
        # linear    num_class
        self.fc_net = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            bn1d(256),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_class)
        )
        # <<< TODO 2.2

    def forward(self, x):
        """
        Define the forward function
        :param x: input features with size [batch_size, in_channels, img_h, img_w]
        :return: output features with size [batch_size, num_classes]
        """
        # Step 1: apply spatial transformer network if applicable
        x0 = self.stn(x)

        # >>> TODO 2.3: forward process
        # Step 2: forward process for the convolutional layers, apply residual connection in conv3 and conv5
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.pool1(x2)
        x4 = self.conv3(x3) + x3
        x5 = self.conv4(x4)
        x6 = self.pool2(x5)
        x7 = self.conv5(x6) + x6

        # Step 3: use `Tensor.view()` to flatten the tensor to match the size of the input of the
        # fully connected layers.
        x7 = x7.view(x7.size(0), -1)
        # Step 4: forward process for the linear layers
        out = self.fc_net(x7)
        # <<< TODO 2.3

        return out


class STN(nn.Module):
    def __init__(self, in_channels):
        """
        The spatial transformer network (STN) learns how to perform spatial transformations on the
        input image in order to enhance the geometric invariance of the model. For example, it can
        crop a region of interest, scale and correct the orientation of an image. It can be a useful
        mechanism because CNNs are not invariant to rotation and scale and more general affine
        transformations.

        The spatial transformer network boils down to three main components:

        - The localization network is a regular CNN which regresses the transformation parameters.
          The transformation is never learned explicitly from this dataset, instead the network
          learns automatically the spatial transformations that enhances the global accuracy.
        - The grid generator generates a grid of coordinates in the input image corresponding
          to each pixel from the output image.
        - The sampler uses the parameters of the transformation and applies it to the input image.

        Here, we are going to implement an STN that performs affine transformations on the input images.
        For more information, please refer to the slides.

        ----------------------
        :param in_channels: channel number of input image
        """
        super().__init__()

        # >>> TODO 3.1: Build your localization net
        # Step 1: Build a convolutional network to extract features from input images.
        # Hint: Combine convolutional layers, batch normalization layers and ReLU activation functions to build
        # this network.
        # Suggested structure: 3 layers of down-sampling convolution (e.g. each layer reduces the feature map
        # size by half), double the number of channels in each layer, use BN and ReLU.
        self.localization_conv = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, 5, stride=1, padding=2),
            nn.BatchNorm2d(2 * in_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(2 * in_channels, 4 * in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(4 * in_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(4 * in_channels, 8 * in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(8 * in_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # Step 2: Build a fully connected network to predict the parameters of affine transformation from
        # the extracted features.
        # Hint: Combine linear layers and ReLU activation functions to build this network.
        # Suggested structure: 2 linear layers with one BN and ReLU.
        self.localization_fc = nn.Sequential(
            nn.Linear(8 * in_channels * 16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        # <<< TODO 3.1

        # >>> TODO 3.2: Initialize the weight/bias of the last linear layer of the fully connected network
        # Hint: The STN should generate the identity transformation by default before training.
        # How to initialize the weight/bias of the last linear layer of the fully connected network to
        # achieve this goal?
        self.localization_fc[3].weight.data.zero_()
        self.localization_fc[3].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
            )
        # <<< TODO 3.2

    def forward(self, x):
        # Extract the features from input images and flatten them
        features = self.localization_conv(x)
        features = features.view(features.shape[0], -1)

        # Predict the parameters of affine transformation from the extracted features
        theta = self.localization_fc(features)
        theta = theta.view(-1, 2, 3)

        # Apply affine transformation to input images
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x
