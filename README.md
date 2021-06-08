# AsppDepth

本算法在MonoDepth2基础上进行开发，整体网络基于pytorch，实现更精准的无监督训练下的单目图像深度估计。

<p align="center">
  <img src="test/image_pre.gif" alt="example input output gif" width="600" />
</p>

<p align="center">
  <img src="test/image_depth.gif" alt="example input output gif" width="600" />
</p>

该代码为本人的work，禁止商业使用，如转载请标明出处，谢谢合作。

### 配置

如果使用了Anaconda，则使用以下指令安装依赖项：

```
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   #评估时使用
```

### 训练与测试

训练和测试方法在与monodepth2流程相同，不过由于网络架构以及代码不同，需要载入您通过本算法训练得到的权重。

#### 数据戳提取：

其中splits.py为提取kitti数据集划分目录脚本，详情参见代码。

#### 训练：

默认情况下，模型和tensorboard文件保存到了

```
~/tmp/<model_name>
```

中，可以使用--log_dir缺省值进行更改。

（1）单目训练：

```
python train.py --model_name mono_model
```

（2）双目训练

```
python train.py --model_name stereo_model \
  --frame_ids 0 --use_stereo --split eigen_full
```

（3）单目+双目训练

```
python train.py --model_name mono+stereo_model \
  --frame_ids 0 -1 1 --use_stereo
```

注：如果您只用单个GPU，可以使用以下指令进行指定：

```
CUDA_VISIBLE_DEVICES=x python train.py --model_name mono_model #其中x为您的device编号
```

#### 测试：

可以使用test_simple.py文件预测单张图像的深度：

```
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192
```

其他细节，可以参考monodepth2中的说明文档：

