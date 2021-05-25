# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

#12.4，over


#file_dir就是option.py文件所在的文件夹
file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="aspp options")

        # PATHS路径
          #训练数据的存放路径，默认的是“./kitti_data/”文件夹
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
          #日志地址默认的是“~/tmp/”.
          #写路径的时候，如果想用"~",要用os.path.expanduser（“~”）,
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "self_train_model_1"))


        # TRAINING options 训练选项
          #存放模型的文件的名称，默认是“mdp：model dump”
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
          #时间戳文件：如何读取数据集中的图片，这里可以设置成自己的文件夹，
          #现在splits文件夹中放入自己的时间戳文件，然后choices选项中加入自己的文件夹名称
          #默认为eigen_zhou
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark","1024"],
                                 default="eigen_zhou")
          #网络层数，默认18层，可以设置高点！
          #设置不同，等训练时调用的model文件里的网络就不同
          #yang：等改完自己的网络，可以加入choices，然后到时候直接控制变量法，做对比！！
        # self.parser.add_argument("--num_layers",
        #                          type=int,
        #                          help="number of resnet layers",
        #                          default=18,
        #                          choices=[18, 34, 50, 101, 152])
          #数据集，选项很多，默认是kitti
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
          #数据集中的图片格式，一般用jpgs格式，但是kitti是png格式的，用下面的指令对其进行调整
          #find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
          #图像的高，默认192
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
          #图像的宽，默认640
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
          #视差平滑度参数，权重，1e-3
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
          #尺度因子数组，默认是0,1,2,3
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
          #最小最大深度，将深度缩放到一定的范围（0.1-100）
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
          #是否使用双目摄像头，双目或单目+双目训练的时候需要set一下
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
          #frame_ids，有两种 一种是（0）。一种是（0，-1,1）
          #nargs="+"代表参数可以设置一个或者多个
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options 优化选项
          #batch_size 每个batch中训练样本的数量，默认12
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
          #学习率、默认1e-4
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
          #epoch代表训练轮数，默认20
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
          #调整程序的步长，默认15，调整学习率用的
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options 切除选项
          #如果设置了，则使用  monodepth v1 多尺度
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
          #使用平均重投影损失
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
          #设置了之后，不会去做auto-masking
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        #   #设置了的话，会采用预测隐蔽方案。
        # self.parser.add_argument("--predictive_mask",
        #                          help="if set, uses a predictive masking scheme as in Zhou et al",
        #                          action="store_true")
          #设置了的话，损失中不会计算ssim
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
          #权重初始化，可以选择是预训练，还是scratch。默认是预训练
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
          #位姿网络一次获取多少图像，默认是一对，可以设置为所有。
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        #   #位姿网络类型，可以选择三种：pose_cnn，独立的残差网络，共享的网络
        # self.parser.add_argument("--pose_model_type",
        #                          type=str,
        #                          help="normal or shared",
        #                          default="separate_resnet",
        #                          choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options系统选项
          #是否使用gpu
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
          #线程数
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options载入选项
          #载入权重文件的名字！！！
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
          #载入模型文件的名字，默认是  encoder采用depth，poseencoder选pose
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options记录选项
          #记录频率，默认250
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
          #保存频率，默认1
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options评估选项
          #双目评估
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
          #单目评估
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
          #评估中不适用中值缩放
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
          #如果设置了，将预测值乘以这个数字，默认是1（预测的深度的尺度因子）
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
          #要计算的.npy差异文件的可选路径
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
          #评估文件的时间戳
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
          #保存预测出来的视差图
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
          #决定是否评估的选项
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
          #如果set假设我们从npy加载eigen的结果，但是我们想用新的基准进行评估。
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
          #如果设置了，将会把视差输出到当前文件夹
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
          #如果set将从原始monodepth 论文里的 ，执行翻转后再处理
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
