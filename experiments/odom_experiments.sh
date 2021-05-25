# A different kitti dataset is required for odometry training and evaluation.
#里程计的训练与评估需要不同的kitti数据集
# This can be downloaded from http://www.cvlibs.net/datasets/kitti/eval_odometry.php
#假设，文件已经被下载到kitti_data_odom
# We assume this has been extraced to the folder ../kitti_data_odom

# Standard mono odometry model.
#标准单目里程计模型
python ../train.py --model_name M_odom \
  --split odom --dataset kitti_odom --data_path ../kitti_data_odom

# Mono odometry model without Imagenet pretraining
#不带有image-net预训练的，单目里程计模型
python ../train.py --model_name M_odom_no_pt \
  --split odom --dataset kitti_odom --data_path ../kitti_data_odom \
  --weights_init scratch --num_epochs 30

# Mono + stereo odometry model
#单目+双目 里程计模型
python ../train.py --model_name MS_odom \
  --split odom --dataset kitti_odom --data_path ../kitti_data_odom \
  --use_stereo

# Mono + stereo odometry model without Imagenet pretraining
#不带有image-net预训练的，单目+双目里程计模型
python ../train.py --model_name MS_odom_no_pt \
  --split odom --dataset kitti_odom --data_path ../kitti_data_odom \
  --use_stereo \
  --weights_init scratch --num_epochs 30
