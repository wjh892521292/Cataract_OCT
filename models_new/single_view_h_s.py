import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from collections import OrderedDict


class ResNet(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.dp = args.dropout
        self.layers = args.resnet_layers

        model = getattr(torchvision.models, 'resnet{}'.format(self.layers))
        self.resnet = model(pretrained=True)

        self.feature_x = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4)

        self.feature_y = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4)

        self.fc = nn.Linear(512, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        # +1 最后维为回归值

        self.top = nn.Linear(128, 1)

        if self.dp < 1.0:
            self.dropout = nn.Dropout(p=self.dp)

    def model_name(self):
        return 'Resnet-{}'.format(self.layers)

    # make some changes to the end layer contrast to the original resnet
    def forward(self, x, y, z):

        x = self.feature_x(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        bcva_det = self.fc(x)
        bcva_det = self.resnet.relu(bcva_det)
        # if self.dp < 1.0:
        #     x = self.dropout(x)
        bcva_det = self.top(bcva_det)

        bcva = self.sigmoid(bcva_det + z) * 1.5
        return bcva, bcva - z, bcva - z - 0.2



