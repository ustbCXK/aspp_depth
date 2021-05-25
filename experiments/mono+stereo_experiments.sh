# Our standard mono+stereo model 
# 标准单目+双目模型
# python ../train.py --model_name MS_640x192 \
#   --use_stereo --frame_ids 0 -1 1

# Our low resolution mono+stereo model
#低分辨率 单目+双目
# python ../train.py --model_name MS_416x128 \
#   --use_stereo --frame_ids 0 -1 1 \
#   --height 128 --width 416

# Our high resolution mono+stereo model
#高分辨率 单目+双目
# python ../train.py --model_name MS_1024x320 \
#   --use_stereo --frame_ids 0 -1 1 \
#   --height 320 --width 1024 \
#   --load_weights_folder ~/tmp/MS_640x192/models/weights_9 \
#   --num_epochs 5 --learning_rate 1e-5

# Our standard mono+stereo model w/o pretraining
#不带预训练的单目+双目
# python ../train.py --model_name MS_640x192_no_pt \
#   --use_stereo --frame_ids 0 -1 1 \
#   --weights_init scratch \
#   --num_epochs 30 \
#   --gpu 1 \
#   --data_path /media/cf/98A8EFD0A8EFAB48/chengxu/aspp_depth_test_1/kitti_data

# Baseline mono+stereo model, i.e. ours with our contributions turned off
#单目+双目的baseline。
# python ../train.py --model_name MS_640x192_baseline \
#   --use_stereo --frame_ids 0 -1 1 \
#   --v1_multiscale --disable_automasking --avg_reprojection \
#   --gpu 1 \
#   --data_path /media/cf/98A8EFD0A8EFAB48/chengxu/aspp_depth_test_1/kitti_data

# Mono+stereo without full-res multiscale
#不带有全分辨率多尺度的单目+双目
# python ../train.py --model_name MS_640x192_no_full_res_ms \
#   --use_stereo --frame_ids 0 -1 1 \
#   --v1_multiscale \
#   --gpu 1 \
#   --data_path /media/cf/98A8EFD0A8EFAB48/chengxu/aspp_depth_test_1/kitti_data

# # Mono+stereo without automasking
# #不带有auto-masking的单目+双目
# python ../train.py --model_name MS_640x192_no_automasking \
#   --use_stereo --frame_ids 0 -1 1 \
#   --disable_automasking \
#   --gpu 1 \
#   --data_path /media/cf/98A8EFD0A8EFAB48/chengxu/aspp_depth_test_1/kitti_data

# # Mono+stereo without min reproj
# # 不带有最小重投影的单目+双目
# python ../train.py --model_name MS_640x192_no_min_reproj \
#   --use_stereo --frame_ids 0 -1 1 \
#   --avg_reprojection  \
#   --gpu 1 \
#   --data_path /media/cf/98A8EFD0A8EFAB48/chengxu/aspp_depth_test_1/kitti_data
