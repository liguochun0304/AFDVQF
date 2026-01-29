STAMP=$(date +%F_%H%M%S)
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_mqspn_original_span_${STAMP}.log"

python3 ../train.py \
  --device "cuda:0" \
  --epochs 50 \
  --batch_size 16 \
  --drop_prob 0.5 \
  --fin_tuning_lr 2e-5 \
  --downs_en_lr 1e-4 \
  --weight_decay_rate 0.01 \
  --clip_grad 1.0 \
  --warmup_prop 0.1 \
  --gradient_accumulation_steps 2 \
  --patience 1e-5 \
  --patience_num 20 \
  --text_encoder "bert" \
  --image_encoder "clip-patch32" \
  --dataset_name "twitter2015" \
  --ex_project "mner_experiment" \
  --ex_name "mqspn_original_span" \
  --ex_nums "A0" \
  --use_image \
  --model "mqspn_original" \
  --decoder_type "span" |& tee -a "$LOG_FILE"
