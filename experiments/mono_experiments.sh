# Our standard mono model
#标准单目训练模型
#python ../train.py --model_name M_640x192

# Our low resolution mono model
#低分辨率单目训练模型
#python ../train.py --model_name M_416x128 \
#  --height 128 --width 416

# Our high resolution mono model
#高分辨率单目训练模型
# python ../train.py --model_name M_1024x320 \
#   --height 320 --width 1024 \
#   --load_weights_folder ~/self_train_model_1/M_640x192/models/weights_9 \
#   --num_epochs 5 --learning_rate 1e-5

# Our standard mono model w/o pretraining
# 标准单目模型（不经过预训练的，w/o是without）
python ../train.py --model_name M_640x192_no_pt \
  --weights_init scratch \
  --num_epochs 30  


# Baseline mono model, i.e. ours with our contributions turned off
#单目模型的baseline，即不使用automasking、使用平均重投影等
#python ../train.py --model_name M_640x192_baseline \
#  --v1_multiscale --disable_automasking --avg_reprojection

# Mono without full-res multiscale
#不带有全分辨率多尺度的单目
python ../train.py --model_name M_640x192_no_full_res_ms \
  --v1_multiscale

# Mono without automasking
#不带有automasking的单目
python ../train.py --model_name M_640x192_no_automasking \
  --disable_automasking

# Mono without min reproj
#不带有最小重投影误差的单目，使用平均重投影误差
python ../train.py --model_name M_640x192_no_min_reproj \
  --avg_reprojection

# Mono with Zhou's masking scheme instead of ours
#用zhou的masking代替我们的auto-masking
#python ../train.py --model_name M_640x192_zhou_masking \
 # --disable_automasking --zhou_mask
