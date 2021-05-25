import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *

#depth网络，是针对单张图进行估计，将resnet_encoder作为depth的encode
        #循环1
        # 拼接+卷积      in:1024+256 x14x14   out:256x14x14

        #循环2
        # 先卷积         in：256x14x14，     out:128x14x14 1
        # 插值上采样，   in：128x14x14，     out：128x28x28
        # 拼接+卷积      in:512+128 x28x28   out:128x28x28

        #循环3
        # 先卷积         in：128x28x28，     out:64x28x28 1
        # 插值上采样，   in：64x28x28，     out：64x56x56
        # 拼接+卷积      in:64+64 x56x56   out:64x56x56

        #循环4
        # 先卷积         in：64x56x56，     out:32x56x56 1
        # 插值上采样，   in：32x56x56，     out：32x112x112
        # 拼接+卷积      in:32+64x112x112   out:32x112x112

        #循环5
        # 先卷积         in：32x112x112，     out:16x112x112 1
        # 插值上采样，   in：16x112x112，     out：16x224x224
        # 拼接+卷积      in:16x224x224       out:16x224x224

class DepthDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales, use_skips=True):
        """
        参数：
        num_ch_enc：encoder网络给到的特征图channels
        scales:尺度
        num_out_channels:最终depth网络输出的通道数
        """
        super(DepthDecoder, self).__init__()

        # 参数传递：
        self.num_out_channels = 1
        self.use_skips = use_skips
        self.unsample_mode = "nearest"
        self.scales = scales
        self.num_ch_enc = np.array([64, 64, 512, 1024, 256])

        # 字典对象排序
        self.convs = OrderedDict()

        # num_ch_enc = np.array([64, 64, 512, 1024, 256])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # 公用层
        self.sigmoid = nn.Sigmoid()

        # 倒着拼接 特征图
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            # conv3x3：填充层加卷积层，最主要是变维度
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]

        x = [x]
        for i in range(4, -1, -1):

            #卷积+上采样
            if i < 4:
                x = self.convs[("upconv", i, 0)](x)
                x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)

            #拼接加卷积（改变通道数）
            x = self.convs[("upconv", i, 1)](x)

            #disp图
            if i in range(4):
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
