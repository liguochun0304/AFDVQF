python3 train.py \
  --dataset_name twitter2015 \
  --device cuda:0 \
  --ex_name ablation_simple_fusion \
  --save_name ablation_simple_fusion_twitter2015 \
  --use_qfnet false \
  --use_simple_fusion true \
  --use_adaptive_fusion false

python3 test.py \
  --save_name ablation_simple_fusion_twitter2015 \
  --device cuda:0 \
  --split test \
  --record_file logs/ablation_records.csv

python3 train.py \
  --dataset_name twitter2017 \
  --device cuda:0 \
  --ex_name ablation_simple_fusion \
  --save_name ablation_simple_fusion_twitter2017 \
  --use_qfnet false \
  --use_simple_fusion true \
  --use_adaptive_fusion false

python3 test.py \
  --save_name ablation_simple_fusion_twitter2017 \
  --device cuda:0 \
  --split test \
  --record_file logs/ablation_records.csv
