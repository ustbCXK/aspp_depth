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
import numpy as np


from collections import OrderedDict

class ResNetMultiImageInput(models.ResNet):
    """
    Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    其实是在构造一个，输入多个图像的resnet模型
    """

    def __init__(self, block, layers, num_input_images=1):
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

        # 权重初始化
        for m in self.modules():
            # isinstance(object, classinfo)，判断某变量object是不是classinfo类型的。
            # 接下来就是对卷积网络进行kaiming_normal 初始化初始化
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_（）函数：权重初始化函数，
                # torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # 对正则化网络进行固定值初始化： torch.nn.init.constant_（）
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(pretrained=False, num_input_images=1):
    """
    Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input

    构造一个ResNet模型：
    参数：
        num_layers（int）：resnet层的数量。必须是18或者50
        pretrained（bool）：如果为True，则返回在ImageNet上预先训练的模型
        num_input_images（int）：堆叠为输入的帧数
    """

    # 残差网络  如果是resnet18的话 block层数就是2，2，2，2；如果是resnet50的话，block层数就是3，4，6，3。
    blocks = [2, 2, 2, 2]
    # block类型：resnet18是BasicBlock，resnet50是Bottleneck
    block_type = models.resnet.BasicBlock
    # 根据选用网络不同，建立不同网络，resnet18或resnet50
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    # 载入预训练权重
    if pretrained:
        # 通过调用model的load_state_dict方法用预训练的模型参数来初始化你构建的网络结构
        # 这个方法就是pytorch中通用的用一个模型的参数初始化另外一个模型的层参数的操作。
        # load_state_dict方法还有一个重要的参数是strict，该参数默认是true，表示预训练模型的层和你的网络结构层严格对应相等。
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet18'])
        # torch.cat（，1），按照维度1进行拼接，就是堆起来。
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class PoseEncoder(nn.Module):
    """
    Pytorch module for a resnet encoder
    """

    def __init__(self, pretrained, num_input_images=1):
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


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        # num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        # num_frames_to_predict_for：要预测的帧数
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        # OrderedDict（）：实现了对字典对象中元素的排序
        # 这个是为了把网络变为列表
        self.convs = OrderedDict()
        # Conv2d（512,256,1）
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        # 位姿0
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        # 位姿1
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        # 位姿2
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        # nn.ModuleList（），它被设计用来存储任意数量的nn. module。
        # ModuleList是Module的子类，当在Module中使用它的时候，就能自动识别为子module。
        # values() 函数以列表返回字典中的所有值。
        # list（）：list() 方法用于将元组转换为列表。元组与列表是非常类似的，区别在于元组的元素值不能修改，元组是放在括号中，列表是放于方括号中。
        self.net = nn.ModuleList(list(self.convs.values()))
        # 至此，self.net()存放了，convs网络的列表形式，用的时候，直接提取就行。

    def forward(self, input_features):
        # last_features是将inputs_features[]，提取最后一个
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        # 比如6x16x16的特征图，经过求平均，变成6x1x1的，其中0维度是batch_size，1维度是6，2、3维度是16
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

if __name__ == "__main__":

    x = np.array([64, 64, 128, 256, 512])
    model = PoseEncoder(pretrained=True, num_input_images=2)
    model.eval()
    print("model is OK")

    model2 = PoseDecoder(num_ch_enc = x, num_input_features = 1,
                         num_frames_to_predict_for = 2)
    model2.eval()
    print("model2 is OK")

    image1 = torch.randn(1, 3, 224, 224)
    image2 = torch.randn(1, 3, 224, 224)
    image3 = torch.randn(1, 3, 224, 224)


    input1 = torch.cat([image1,image2], 1)
    input11 = torch.cat([image2, image1], 1)
    print("size of input1:", input1.size())

    input2 = torch.cat([image2,image3], 1)
    print("size of input2:", input2.size())

    input3 = torch.cat([image1,image3], 1)
    print("size of input3:", input3.size())

    outputs = {}
    with torch.no_grad():

        output1 = [model.forward(input1)]
        axisangle, translation = model2.forward(output1)
        print("axisangle:\n",axisangle)
        print("size of axisangle:",axisangle.size())
        print("translation:\n",translation)
        print("size of translation:",translation.size())
        outputs[("axisangle", 1, 2)] = axisangle
        outputs[("translation", 1, 2)] = translation
        outputs[("cam_T_cam", 1, 2)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
        a = outputs[("cam_T_cam", 1, 2)]
        print("cam_T_cam1:",outputs[("cam_T_cam", 1, 2)])
        print("size of cam_T_cam1:", outputs[("cam_T_cam", 1, 2)].size())

        output1 = [model.forward(input1)]
        axisangle, translation = model2.forward(output1)
        print("axisangle:\n", axisangle)
        print("size of axisangle:", axisangle.size())
        print("translation:\n", translation)
        print("size of translation:", translation.size())
        outputs[("axisangle", 1, 2)] = axisangle
        outputs[("translation", 1, 2)] = translation
        outputs[("cam_T_cam", 1, 2)] = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=True)
        a = outputs[("cam_T_cam", 1, 2)]
        print("cam_T_cam1:", outputs[("cam_T_cam", 1, 2)])
        print("size of cam_T_cam1:", outputs[("cam_T_cam", 1, 2)].size())

        output11 = [model.forward(input11)]
        axisangle, translation = model2.forward(output11)
        print("axisangle:\n", axisangle)
        print("size of axisangle:", axisangle.size())
        print("translation:\n", translation)
        print("size of translation:", translation.size())
        outputs[("axisangle", 2, 1)] = axisangle
        outputs[("translation", 2, 1)] = translation
        outputs[("cam_T_cam", 2, 1)] = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=False)
        a = outputs[("cam_T_cam", 2, 1)]
        print("cam_T_cam2:", outputs[("cam_T_cam", 2, 1)])
        print("size of cam_T_cam1:", outputs[("cam_T_cam", 2, 1)].size())


        loss = outputs[("cam_T_cam", 2, 1)] * outputs[("cam_T_cam", 1, 2)]
        print(loss)

        # output2 = [model.forward(input2)]
        # axisangle, translation = model2.forward(output2)
        # print("axisangle:\n",axisangle)
        # print("size of axisangle:",axisangle.size())
        # print("translation:\n",translation)
        # print("size of translation:",translation.size())
        # outputs[("axisangle", 2, 3)] = axisangle
        # outputs[("translation", 2, 3)] = translation
        # outputs[("cam_T_cam", 2, 3)] = transformation_from_parameters(
        #                 axisangle[:, 0], translation[:, 0], invert=False)
        # b = outputs[("cam_T_cam", 2, 3)]
        # print("cam_T_cam2:",outputs[("cam_T_cam", 2, 3)])
        # print("size of cam_T_cam2:", outputs[("cam_T_cam", 2, 3)].size())
        #
        # output3 = [model.forward(input3)]
        # axisangle, translation = model2.forward(output3)
        # print("axisangle:\n", axisangle)
        # print("size of axisangle:", axisangle.size())
        # print("translation:\n", translation)
        # print("size of translation:", translation.size())
        # outputs[("axisangle", 1, 3)] = axisangle
        # outputs[("translation", 1, 3)] = translation
        # outputs[("cam_T_cam", 1, 3)] = transformation_from_parameters(
        #     axisangle[:, 0], translation[:, 0], invert=True)
        # c = outputs[("cam_T_cam", 1, 3)]
        # print("cam_T_cam3:", outputs[("cam_T_cam", 1, 3)])
        # print("size of cam_T_cam3:", outputs[("cam_T_cam", 1, 3)].size())
        # d = a * b * c
        # print("a * b * c:\n",d)
        # print("size of a * b * c :",d.size())
        # d = d.mean(0, keepdim = False)
        # print("(a * b * c).mean():\n", d)
        # print("size of (a * b * c).mean() :", d.size())
        #
        #
        # I = np.identity(4)
        # P = np.identity(4)
        # print("I:\n",I)
        # loss = P - I
        # print("LOSS：", loss)
        # loss = loss.mean()
        # print("LOSS：",loss)

