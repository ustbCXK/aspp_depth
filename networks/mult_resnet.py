# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """
    Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    其实是在构造一个，输入多个图像的resnet模型
    """
    def __init__(self, block, layers,  num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #权重初始化
        for m in self.modules():
            #isinstance(object, classinfo)，判断某变量object是不是classinfo类型的。
            #接下来就是对卷积网络进行kaiming_normal 初始化初始化
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_（）函数：权重初始化函数，
                # torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #对正则化网络进行固定值初始化： torch.nn.init.constant_（）
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(pretrained=False, num_input_images=1):
    """
    构造一个ResNet模型：
    参数：
        num_layers（int）：resnet层的数量。必须是18或者50
        pretrained（bool）：如果为True，则返回在ImageNet上预先训练的模型
        num_input_images（int）：堆叠为输入的帧数   
    """
   
    #残差网络  如果是resnet18的话 block层数就是2，2，2，2；如果是resnet50的话，block层数就是3，4，6，3。
    blocks = [2, 2, 2, 2]
    #block类型：resnet18是BasicBlock，resnet50是Bottleneck
    block_type = models.resnet.BasicBlock
    #根据选用网络不同，建立不同网络，resnet18或resnet50
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    #载入预训练权重
    if pretrained:
        #通过调用model的load_state_dict方法用预训练的模型参数来初始化你构建的网络结构
        #这个方法就是pytorch中通用的用一个模型的参数初始化另外一个模型的层参数的操作。
        #load_state_dict方法还有一个重要的参数是strict，该参数默认是true，表示预训练模型的层和你的网络结构层严格对应相等。
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet18'])
        #torch.cat（，1），按照维度1进行拼接，就是堆起来。
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class PoseEncoder(nn.Module):
    """
    Pytorch module for a resnet encoder
    """
    def __init__(self,  pretrained, num_input_images=1):
        super(PoseEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])


        self.encoder = resnet_multiimage_input(pretrained, num_input_images)
        
    def forward(self, input_image):

        self.features = []

        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

