"""
本版本为基础版本去掉了回环一致性

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from layers import *
from kitti_utils import *

import datasets
import networks
from IPython import embed
#多GPU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"


#采用GPU1
torch.cuda.set_device(1) 

class Trainer:

    """
    初始化：
    1、基础参数配置
    2、网络设定
    3、优化器\载入网络初始权重
    4、载入数据
    5、网络细节设定，损失函数如何计算
    """
    def __init__(self, options):

        #1、基础参数配置
        self.opt = options
        #日志保存路径："log_dir/model_name" 默认的是“~/tmp/mdp”
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height'必须是32的倍数"
        assert self.opt.width % 32 == 0, "'width'必须是32的倍数"

        #model字典：一会儿会存入depth网络和pose网络的encoder和decoder
        self.models = {}

        #存放训练过程中 需要优化的参数，方便adam优化器优化
        self.parameters_to_train = []

        #默认采用GPU加速
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        #多尺度scales参数设置，损失loss计算会用到
        # self.opt.scales：default = [0, 1, 2, 3]
        # num_scales = 4
        self.num_scales = len(self.opt.scales)

        # 比如frame_ids= [0]或者frame_ids = [0,-1,1]
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        #输入帧的数量:1 or 3
        self.num_input_frames = len(self.opt.frame_ids)

        #需要修改：只要成对输入
        #输入pose网络的帧的数目，由model_input决定，
        #pose网络encoder的输入维度:num_pose_frames * 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        #是否采用位姿，看是否只输入当前帧
        self.use_pose_net = not ( self.opt.use_stereo and self.opt.frame_ids == [0] )

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        #2、网络设定
        #包括depth网络，pose网络
        #   1、depth网络
        #      models["encoder"]        ResnetEncoder
        #      models["depth"]:         DepthDecoder          
        #   2、pose网络
        #      models["pose_encoder"]:  PoseDecoder
        #      models["pose"]:          PoseEncoder
       
        #encoder采用resnetEncoder，直接采用50层
        #__init__(self, nInputChannels, block = Bottleneck, pretrained=False):
        self.models["encoder"] = networks.ResnetEncoder(
            nInputChannels = 3, pretrained = self.opt.weights_init)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        #decoder采用depthdecoder,输入维度为encoder的输出维度,scales与options保持一致
        #def __init__(self, num_ch_enc, scales, use_skips=True)
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())


        #存疑，为什么pose_encoder参数num_frames_to_predict_for = 2
        #如果使用pose_ent（单目或者单目+双目）
        if self.use_pose_net:
            #__init__(self,  pretrained, num_input_images=1):
            self.models["pose_encoder"] = networks.PoseEncoder(
                pretrained = self.opt.weights_init,
                num_input_images = self.num_pose_frames)
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            #__init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features = 1,
                num_frames_to_predict_for = 2)
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
        
        #3、优化器
        #优化器采用adam
        #learning_rate 默认：1e-4
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        #scheduler_step_size 默认：15
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        #如果需要训练高精度，则用到该参数
        if self.opt.load_weights_folder is not None:
            self.load_model()

        #self.opt.model_name给训练的模型  起的名字
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)


        #4、载入数据:
        #（1）载入时间戳文件
        datasets_dict = {
            "kitti":datasets.KITTIRAWDataset,
            "kitti_odom":datasets.KITTIOdomDataset}
        #self.opt.dataset 默认：default="kitti"
        self.datasets = datasets_dict[self.opt.dataset]

        #读取splits文件,self.opt.splits：train或者val
        #读取路径："当前文件夹/splits/xx_files.txt"
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        #batch_size = 12 ; num_epochs = 20
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs


        #（2）根据时间戳文件，读取数据
        train_dataset = self.datasets(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, num_scales = 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers = self.opt.num_workers, pin_memory = True, drop_last = True)

        val_dataset = self.datasets(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, num_scales = 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers = self.opt.num_workers, pin_memory = True, drop_last = True)
        #产生一个迭代器，验证过程会用到
        self.val_iter = iter(self.val_loader)

        #创建一个字典，包含train和val,用来保存日志
        self.writers = {}
        for mode in ["train","val"]:
            #log()会用
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path ,mode))

        #5、网络细节设定，损失函数如何计算
        #（1）关于SSIM（）
        #（2）逆投影深度
        #（3）深度图投影出3D点
        #（4）各项指标

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        #重投影的空间点字典
        self.backproject_depth = {}

        #深度图投影3D点
        self.project_3d = {}

        #scales ： （0，,1,2,）
        #计算每一个尺度的图像的重投影（深度图到3D点）以及投影（3D点到深度图）
        for scale in self.opt.scales:
            h = self.opt.height //(2 ** scale)
            w = self.opt.width //(2 ** scale)

            #BackprojectDepth()在layers里
            #( batch_size, height, width)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            #将3D点投影到深度图
            #(batch_size, height, width, eps=1e-7)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using splits:\n", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))      

        self.save_opts()

    #训练函数
    def train(self):
        """
        整个训练过程，就是跑n个epochs
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.opt.num_epochs):

            self.run_epoch()
            if (self.epoch + 1 ) % self.opt.save_frequency == 0:
                self.save_model()


    #单独的运行一个epoch
    def run_epoch(self):
        """
        一个epoch，跑12个mini_batch_size
        """

        self.model_lr_scheduler.step()

        print("training:")

        #将模型设置为训练模式
        self.set_train()

        #计算一个epoch的outputs和losses
        for batch_idx, inputs in enumerate(self.train_loader):
        
            #开始计时:
            before_op_time = time.time()

            # inputs：一个一个的batch，dataloader划分好的
            # return：(1)output[("depth",0,scales),("sample",frame_id, scale),("color", frame_id, scale),
            #                   ("color_identity", frame_id, scale)]
            #        (2)losses[("loss"),("loss\0"),("loss\1"),("loss\2"),("loss\3")]
            outputs, losses = self.process_batch(inputs)

            #执行优化
            self.model_optimizer.zero_grad()

            #平均误差进行反向传播
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            #2000步后，降低记录频率
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                #单纯的计算一下，现在loss 不做backward
                if "depth_gt" in inputs:
                    #计算深度losses，以便在培训期间进行监控
                    print("calculating:compute_depth_losses:")                    
                    self.compute_depth_losses(inputs, outputs, losses)

                # 到此，losses中多了abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

                self.log("train", inputs, outputs, losses)
                #验证
                self.val()

            self.step += 1


    #将载入的所有模型，设置为训练模式
    def set_train(self):
        #比如depth网络采用depth_encodet+depth_decoder;pose网络采用了pose_cnn
        #.values()进行遍历
        for m in self.models.values():
            m.train()


    #将载入的所有模型，设置为验证模式
    def set_eval(self):
        """
        Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()


    #对输入进行一次运算
    #返回： (1)output[[("disp",i)],("depth",0,scales),("sample",frame_id, scale),("color", frame_id, scale),
    #                   ("color_identity", frame_id, scale)]
    #      (2)losses[("loss"),("loss\0"),("loss\1"),("loss\2"),("loss\3")]
    def process_batch(self, inputs):
        """
        inputs:
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width     , self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        #items() 函数：以列表返回可遍历的(键, 值) 元组数组
        #接下来，inputs[key],就代表了kitti数据
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # depth网络
        #   1、depth_encoder:inputs：inputs["color_aug", 0, 0]，其中(frame_id=0, scale = 0)
        #                    return:features[0,1,2,3,4]
        #   2、depth_decoder:inputs：features[0,1,2,3,4]
        #                    return：outputs[("disp", 0),("disp", 1),("disp", 2),("disp", 3)]
        #结果：outputs[("disp",i)]
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        # #yxp: depth网络计算I（t+1）的深度图D（t+1）
        # #outputs2["disp",i]    (It+1)的视差图
        # features2 = self.models["encoder"](inputs["color_aug", 1, 0])
        # outputs2 = self.models["depth"](features2)

        # for i in range(4):
        #     outputs[("disp2",i)] = outputs2[("disp",i)]
        # #yxp:

        #predict_poses()：pose网络
        #   inputs: inputs[], features[]
        #   return: outputs["axisangle","translation","cam_T_cam"]
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs))



        #到此，outputs["disp","disp2","axisangle","translation","cam_T_cam"]

        # 产生预测图
        # 重投影过程，生成新的目标图像
        # inputs：（inputs，outputs）
        # return： output[("depth",0,scale),("sample",frame_id, scale),("color", frame_id, scale),
        #                   ("color_identity", frame_id, scale)]
        #       附加：output[("depth2",1,scale),("sample2",-1,scale),("color2",-1,scale),
        #                   ("color_identity2",-1,scale)]
        self.generate_images_pred(inputs, outputs)


        #结合inputs[]、和outputs[]计算losses
        #inputs： (inputs[], ouputs[])
        #return: losses[("loss"),("loss\0"),("loss\1"),("loss\2"),("loss\3")]
        losses = self.compute_losses(inputs,outputs)


        return outputs, losses


    #预测frame_ids之间的位姿
    #输入：inputs
    #输出：pose：(axisangle，translation):(-1,0), (0,1), (1,-1)
    #           (cam_T_cam):(-1,0)-1, (0,1), (1,-1)
    def predict_poses(self, inputs):
        """
        估计两帧之间的pose
        """

        outputs = {}

        #以frame_ids = (0,-1,1,s)为例
        #pose_feats = {'0':inputs["color_aug,"],'0','0',
        #               '-1':inputs["color_aug,"],'-1','0',
        #               '1':inputs["color_aug,"],'1','0',
        #               's':inputs["color_aug,"],'s','0',
        #               }
        pose_feats = { f_i:inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids }

        #以f_i作为迭代器,只要(-1,1,s)
        #计算(-1,0)和(0,1)之间的角度和旋转轴
        for f_i in self.opt.frame_ids[1:]:
            #只对[-1,1进行处理]
            if f_i != "s":
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i],pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0],pose_feats[f_i]]
                
                
                #pose_encoder:
                # inputs: pose_inputs(Its type is tensor.)
                # outpus: features[0,1,2,3,4]
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs,1))]

                #pose_decoder:
                # inputs: pose_inputs
                # outpus: axisangle(shape:(1,2,1,3)), translation(shape:(1,2,1,3))
                axisangle, translation = self.models["pose"](pose_inputs)

                #outputs["axisangle","translation","cam_T_cam"]
                outputs[("axisangle",0,f_i)] = axisangle
                outputs[("translation",0,f_i)] = translation

                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[: , 0], translation[: , 0], invert=(f_i < 0))
        
        return outputs


    #在minibatch上验证模型，一般是训练完调用
    def val(self):
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()
        
        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


    #产生预测图
    #重投影过程，生成新的目标图像
    # inputs：（inputs，outputs）
    # return： output[("depth",0,scales),("sample",frame_id, scale),("color", frame_id, scale),
    #                   ("color_identity", frame_id, scale)]
    #       附加：output[("depth2",1,scale),("sample2",-1,scale),("color2",-1,scale),
    #                   ("color_identity2",-1,scale)]
    def generate_images_pred(self, inputs, outputs):
        """
        产生warped（重投影），其实就是将深度点重投影到平面上，产生预测图
        """
        #self.opt.scales:[0,1,2,3]
        for scale in self.opt.scales:

            disp = outputs[("disp", scale)]
            # yxp:
            # disp2 = outputs[("disp2", scale)]
            #如果采用monodepth1里的多尺度，尺度直接采用scale
            if self.opt.v1_multiscale:
                source_scale = scale
            #否则，直接用0
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width],mode = "bilinear",align_corners=False)
                # yxp:
                # disp2 = F.interpolate(
                #     disp2, [self.opt.height, self.opt.width],mode = "bilinear",align_corners=False)
                source_scale = 0

            #从视差图转换为深度预测，
            #disp_to_depth（）返回值：scaled_disp,depth
            _,depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            # yxp:
            # _,depth2 = disp_to_depth(disp2, self.opt.min_depth, self.opt.max_depth)

            #将当前scale下的depth估计出来，并记录到outputs中
            outputs[("depth", 0, scale)] = depth
            # yxp:
            # outputs[("depth2"), 1, scale] = depth2

            #depth为It的深度图Dt，depth2为It+1的深度图
            #1、将Dt  warp到t—1、t+1、R上去
            for i, frame_id in enumerate( self.opt.frame_ids[1: ]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                #根据深度图，计算空间点
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                #根据空间点，投影，生成预测像素
                pix_coords = self.project_3d[source_scale](
                    cam_points,inputs[("K", source_scale)],T)

                #根据预测的像素，swarp到图像上
                outputs[("sample", frame_id, scale)] = pix_coords
                #插值，生成预测图，pred
                # 简单来说就是，提供一个input的Tensor以及一个对应的flow-field网格(比如光流，体素流等)，
                # 然后根据grid中每个位置提供的坐标信息(这里指input中pixel的坐标)，
                # 将input中对应位置的像素值填充到grid指定的位置，得到最终的输出。
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color",frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
            # yxp:
            #2、将Dt+1  warp得到It-1
            # T2 = outputs[("cam_T_cam", 1,-1)]
            # cam_points2 = self.backproject_depth[source_scale](
            #         depth2, inputs[("inv_K", source_scale)])
            # pix_coords2 = self.project_3d[source_scale](
            #     cam_points2, inputs[("K", source_scale)], T2)
            # outputs[("sample2", -1, scale)] = pix_coords2
            # outputs[("color2", -1, scale)] = F.grid_sample(
            #     inputs[("color", -1, source_scale)],
            #     outputs[("sample", -1, scale)],
            #     padding_mode="border")
            # if not self.opt.disable_automasking:
            #     outputs[("color_identity2", -1, scale)] = \
            #         inputs[("color", -1, source_scale)]
            # yxp:

        

    #计算一批预测图像和目标图像之间的重投影损失
    def compute_reprojection_loss(self, pred, target):
        #误差绝对值
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1,True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss

        else:
            ssim_loss = self.ssim(pred, target).mean(1,True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


    #计算一个小批量的重投影和平滑损失
    def compute_losses(self, inputs, outputs):
        """
        automasking--(1)avg_reprojection
                    |__(2)no avg_reprojection
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp",scale)]
            # yxp:
            # disp2 = outputs[("disp2",scale)]


            color = inputs[("color",0,scale)]
            #yxp
            # color2 =inputs[("color", -1 ,scale)]

            target = inputs[("color",0,source_scale)]
            # yxp:
            # target2 = inputs[("color", 1, source_scale)]

            for frame_id in self.opt.frame_ids[1: ]:
                pred = outputs[("color", frame_id, scale)]
                #重投影误差[]，包括了计算预测图与目标图之间的loss
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            # yxp:
            # pred2 = outputs[("color2", -1 ,scale)]
            # yxp:
            # reprojection_losses.append(self.compute_reprojection_loss(pred2, target2))

            #将各帧之间的误差进行拼接
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    #outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]
                    pred = inputs[("color", frame_id, source_scale)]
                    # yxp:
                    # pred2 = inputs[("color", -1, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred,target))
                    #yxp:
                    # identity_reprojection_losses.append(
                    #     self.compute_reprojection_loss(pred2, target2))
                
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                identity_reprojection_losses += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001
                
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1 )
            else:
                combined = reprojection_loss

            #shape[1]==1，就等于combined = reprojection_loss  就是，没用automasking
            #else， 使用了automasking
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            
            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()


            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            #yxp
            # mean_disp2 = disp2.mean(2,True).mean(3,True)
            # norm_disp2 = disp / (mean_disp2 + 1e-7)
            # smooth_loss2 = get_smooth_loss(norm_disp2, color2)

            #smooth_loss = (smooth_loss + smooth_loss2) / 2


            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses


    #计算回环一致性误差，即

    #计算深度losses，仅仅在培训期间进行监视。，而且还是在输入中包含"depth_gt"时，才用到
    def compute_depth_losses(self, inputs, outputs,losses):
        depth_pred = outputs[("depth", 0, 0)]      

        depth_pred = torch.clamp(F.interpolate(depth_pred,[375,1242], 
            mode="bilinear", align_corners=False),1e-3, 80)

        #梯度分割，
        depth_pred = depth_pred.detach()

        #输入中有depth_gt：一张深度图，根据点云数据生成的groundtruth
        depth_gt = inputs["depth_gt"]
        mask = depth_gt >0

        # garg/eigen crop 裁剪
        #b = torch.zeros_like(a) ：产生一个与a相同shape的全零Tensor
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        #压缩
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        #计算深度误差
        #    depth_errors =  abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            #枚举字典里每一个指标，然后分别计算对应的误差。
            #让各个指标归位
            losses[metric] = np.array(depth_errors[i].cpu())


    
    #记录时间
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))



    #将事件写入tensorboard事件文件
    def log(self, mode, inputs, outputs, losses):
        """
        Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)


    #知道自己本次运行采用的什么配置
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)


    #保存模型
    def save_model(self):
        """
        Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        print("save model in : \n", save_path)
        torch.save(self.model_optimizer.state_dict(), save_path)


    #从硬盘进行模型载入
    def load_model(self):
        """
        Load model(s) from disk
        """
        #load_weights_folder ：model_name，设置为模型的名字，便于载入
        #如果你要用~，你就应该用这个os.path.expanduser把~展开．
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        #isdir：判断是否为一个目录
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        #models_to_load：载入模型文件的名字，默认是  encoder采用depth，poseencoder选pose
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            #model中存放的是需要优化的参数。
            #通过update进行载入。
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        #  载入adam状态
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
