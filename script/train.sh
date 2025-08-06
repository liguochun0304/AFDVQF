python ../train.py \
  --device "cuda:0" \
  --epochs 100 \
  --batch_size 64 \
  --drop_prob 0.3 \
  --downs_en_lr 3e-4 \
  --weight_decay_rate 0.01 \
  --clip_grad 2.0 \
  --warmup_prop 0.1 \
  --gradient_accumulation_steps 2 \
  --min_epoch_num 5 \
  --patience 0.00001 \
  --patience_num 20 \
  --text_encoder "roberta-base" \
  --image_encoder "clip-patch32" \
  --dataset_name "twitter2017" \
  --ex_project "MNER_Baseline_Comparison" \
  --ex_name "test_use_adapter" \
  --ex_nums 0 \
  --use_coattention \
  --fusion_type \
  --use_image
#  --use_bilstm


#python ../train.py \
#  --device "cuda:0" \
#  --epochs 100 \
#  --batch_size 64 \
#  --drop_prob 0.3 \
#  --downs_en_lr 3e-4 \
#  --weight_decay_rate 0.01 \
#  --clip_grad 2.0 \
#  --warmup_prop 0.1 \
#  --gradient_accumulation_steps 2 \
#  --min_epoch_num 5 \
#  --patience 0.00001 \
#  --patience_num 20 \
#  --text_encoder "bert-base-uncased" \
#  --image_encoder "clip-patch32" \
#  --dataset_name "twitter2015" \
#  --ex_project "MNER_Baseline_Comparison" \
#  --ex_name "test_bert_BiLSTM_CRF" \
#  --ex_nums 0