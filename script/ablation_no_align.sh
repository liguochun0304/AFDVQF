python3 train.py \
  --dataset_name twitter2015 \
  --device cuda:0 \
  --ex_name ablation_no_align \
  --save_name ablation_no_align_twitter2015 \
  --use_alignment_loss false

python3 test.py \
  --save_name ablation_no_align_twitter2015 \
  --device cuda:0 \
  --split test \
  --record_file logs/ablation_records.csv

python3 train.py \
  --dataset_name twitter2017 \
  --device cuda:0 \
  --ex_name ablation_no_align \
  --save_name ablation_no_align_twitter2017 \
  --use_alignment_loss false

python3 test.py \
  --save_name ablation_no_align_twitter2017 \
  --device cuda:0 \
  --split test \
  --record_file logs/ablation_records.csv
