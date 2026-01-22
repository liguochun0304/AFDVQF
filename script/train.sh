STAMP=$(date +%F_%H%M%S)
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${STAMP}.log"

python3 ../train.py \
  --device "cuda:0" \
  --epochs 50 \
  --batch_size 64 \
  --drop_prob 0.1 \
  --clip_lr 1e-5 \
  --fin_tuning_lr 5e-5 \
  --downs_en_lr 1.2e-4 \
  --weight_decay_rate 0.01 \
  --clip_grad 1.0 \
  --warmup_prop 0.08 \
  --gradient_accumulation_steps 2 \
  --patience 1e-5 \
  --patience_num 20 \
  --text_encoder "bert" \
  --image_encoder "clip-patch32" \
  --dataset_name "twitter2015" \
  --ex_project "mner_experiment" \
  --ex_name "use_images_model" \
  --ex_nums "A0" \
  --use_bilstm \
  --use_image \
  --contrastive_lambda 0.1 \
  --num_interaction_layers 4 \
  --num_queries 8 \
  --use_dynamic_routing \
  --vision_trainable |& tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
/usr/bin/shutdown -h now
exit $rc