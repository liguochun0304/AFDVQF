python3 train.py \
  --dataset_name twitter2015 \
  --device cuda:0 \
  --ex_name ablation_clip_only \
  --save_name ablation_clip_only_twitter2015 \
  --use_dual_vision_extractor false \
  --use_image true \
  --use_patch_tokens true \
  --use_region_tokens false

python3 test.py \
  --save_name ablation_clip_only_twitter2015 \
  --device cuda:0 \
  --split test \
  --record_file logs/ablation_records.csv

python3 train.py \
  --dataset_name twitter2017 \
  --device cuda:0 \
  --ex_name ablation_clip_only \
  --save_name ablation_clip_only_twitter2017 \
  --use_dual_vision_extractor false \
  --use_image true \
  --use_patch_tokens true \
  --use_region_tokens false

python3 test.py \
  --save_name ablation_clip_only_twitter2017 \
  --device cuda:0 \
  --split test \
  --record_file logs/ablation_records.csv
