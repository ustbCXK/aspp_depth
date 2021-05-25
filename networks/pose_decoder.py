# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()
        
        #num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        #num_frames_to_predict_for：要预测的帧数
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        #OrderedDict（）：实现了对字典对象中元素的排序
        #这个是为了把网络变为ModuleList
        self.convs = OrderedDict()
        #Conv2d（512,256,1）
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        #位姿0
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        #位姿1
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        #位姿2
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        #nn.ModuleList（），它被设计用来存储任意数量的nn. module。
        #ModuleList是Module的子类，当在Module中使用它的时候，就能自动识别为子module。
        #values() 函数以列表返回字典中的所有值。
        #list（）：list() 方法用于将元组转换为列表。元组与列表是非常类似的，区别在于元组的元素值不能修改，元组是放在括号中，列表是放于方括号中。
        self.net = nn.ModuleList(list(self.convs.values()))
        #至此，self.net()存放了，convs网络的列表形式，用的时候，直接提取就行。

    def forward(self, input_features):
        #last_features是将特征图最后一列作为一个矩阵（最后一个特征图，econv5）
        last_features = [f[-1] for f in input_features]


        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        #比如6x16x16的特征图，经过求平均，变成6x1x1的，其中0维度是batch_size，1维度是6，2、3维度是16
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
