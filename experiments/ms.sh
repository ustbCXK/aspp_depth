#高分辨率
python ../train.py --model_name MS_1024x320 \
  --use_stereo --frame_ids 0 -1 1 \
  --height 320 --width 1024 \
  --load_weights_folder ~/self_train_model/test/models/weights_9 \
  --num_epochs 5 --learning_rate 1e-5