#!/usr/bin/env bash
# 两阶段连续训练：S1 打地基 + S2 多轮细调（≥100 epochs/组合）
# S2 每轮从上一轮最优权重继续（--continue_train_name）
# 运行：bash run_roberta_clip_grid_2stage_100ep.sh

set -u  # 不用 -e/pipefail，单个失败不影响整体

PY=python3
TRAIN=../train.py

# ---------- 固定公共超参 ----------
DEVICE="cuda:0"
BATCH=64
DROP=0.25
WD=0.01
GAS=2
MIN_EPOCH=5
PATIENCE=1e-5
PATIENCE_NUM=20

EX_PROJECT="mner_experiment"
MODEL_NAME="MNER"
TEXT_ENCODER="roberta-base"
IMAGE_ENCODER="clip-patch32"

# ---------- 网格维度 ----------
DATASETS=("twitter2015" "twitter2017")
ALIGNS=("0" "0.05" "0.1" "0.2" "0.4" "0.8")
RESAMP_TOKENS=("4" "8")
CLIP_GRADS=("1.0")

# ---------- S1/S2 每数据集的 epoch 规划（确保总计>=100） ----------
S1_EPOCHS_2015=8
S2_EPOCHS_2015=(40 30 22)   # sum=92 → 8+92=100
S1_EPOCHS_2017=6
S2_EPOCHS_2017=(40 30 24)   # sum=94 → 6+94=100

# ---------- LR 规划 ----------
# S1：文本/下游 LR（基准），按 align 缩放下游 LR
FIN_TUNING_LR_2015_S1=4e-5
FIN_TUNING_LR_2017_S1=5e-5
DOWNS_EN_LR_2015_S1=1e-4
DOWNS_EN_LR_2017_S1=1.2e-4

# S2：视觉塔 LR（阶段解冻后使用，固定值）；文本/下游在 S2 轮次中再衰减
CLIP_LR_2015_S2=5e-6
CLIP_LR_2017_S2=1e-5

# 学习率衰减与 warmup（按 S2 轮：r1,r2,r3）
S2_DECAYS=("1.0" "0.6" "0.4")
S2_WARMUPS=("0.04" "0.03" "0.02")

# warmup（S1）
WARMUP_2015_S1=0.06
WARMUP_2017_S1=0.08

DEFAULT_TIMEOUT=""  # 如需防卡死可设 "12h"

# ---------- 日志 ----------
STAMP=$(date +%F_%H%M%S)
DATE_TAG=$(date +%F)   # <<< 新增：一次取当天日期，用于拼 swanlab_name
LOG_DIR="logs_${STAMP}"
mkdir -p "$LOG_DIR"
SUCCESS_LIST="$LOG_DIR/success.list"
FAIL_LIST="$LOG_DIR/fail.list"
: > "$SUCCESS_LIST"
: > "$FAIL_LIST"


# ---------- 工具函数 ----------
float_mul () { awk -v a="$1" -v b="$2" 'BEGIN{printf("%.8g", a*b)}'; }
sci () { printf "%.0e" "$1"; }

scale_by_align () {
  case "$1" in
    0) echo "1.00" ;;
    0.05) echo "0.90" ;;
    0.1) echo "0.90" ;;
    0.2) echo "0.80" ;;
    0.4) echo "0.70" ;;
    0.8) echo "0.60" ;;
    *) echo "1.00" ;;
  esac
}

run_job () {
  local stage="$1"; shift
  local exname="$1"; shift
  local timeout_val="$1"; shift
  local log_file="$LOG_DIR/${exname}.log"

  echo "[RUN] ${stage}: ${exname}"
  echo "Log: ${log_file}"

  if [[ -n "$timeout_val" ]]; then
    timeout "$timeout_val" "$PY" "$TRAIN" "$@" |& tee "$log_file"
  else
    "$PY" "$TRAIN" "$@" |& tee "$log_file"
  fi
  local code=${PIPESTATUS[0]}

  if [[ $code -eq 0 ]]; then
    echo "[OK] ${stage}: ${exname}" | tee -a "$SUCCESS_LIST"
  else
    echo "[FAIL:${code}] ${stage}: ${exname}" | tee -a "$FAIL_LIST"
  fi
  echo
  return $code
}

# 计算总任务数（S1 + S2轮数）
calc_total () {
  local rounds=${#S2_DECAYS[@]}
  echo $(( ${#DATASETS[@]} * ${#ALIGNS[@]} * ${#RESAMP_TOKENS[@]} * ${#CLIP_GRADS[@]} * (1 + rounds) ))
}
TOTAL=$(calc_total)
COUNT=0

# ---------- 主循环 ----------
for align in "${ALIGNS[@]}"; do
  case "$align" in
    0) EX_NUM="B1" ;; 0.05) EX_NUM="B2" ;; 0.1) EX_NUM="B3" ;;
    0.2) EX_NUM="B4" ;; 0.4) EX_NUM="B5" ;; 0.8) EX_NUM="B6" ;;
    *) EX_NUM="BX" ;;
  esac
  SCALE=$(scale_by_align "$align")

  for ds in "${DATASETS[@]}"; do

    # 数据集特定配置
    if [[ "$ds" == "twitter2015" ]]; then
      S1_EPOCHS=$S1_EPOCHS_2015
      S2_EPOCHS_ARR=("${S2_EPOCHS_2015[@]}")
      FIN_TUNING_LR_S1=$FIN_TUNING_LR_2015_S1
      DOWNS_EN_LR_BASE_S1=$DOWNS_EN_LR_2015_S1
      CLIP_LR_S2=$CLIP_LR_2015_S2
      WARMUP_S1=$WARMUP_2015_S1
    else
      S1_EPOCHS=$S1_EPOCHS_2017
      S2_EPOCHS_ARR=("${S2_EPOCHS_2017[@]}")
      FIN_TUNING_LR_S1=$FIN_TUNING_LR_2017_S1
      DOWNS_EN_LR_BASE_S1=$DOWNS_EN_LR_2017_S1
      CLIP_LR_S2=$CLIP_LR_2017_S2
      WARMUP_S1=$WARMUP_2017_S1
    fi

    # 按 align 缩放下游 LR
    DOWNS_EN_LR_S1=$(float_mul "$DOWNS_EN_LR_BASE_S1" "$SCALE")

    # S2 第一轮的基准 LR（之后每轮按 S2_DECAYS 再衰减）
    FIN_TUNING_LR_S2_BASE=$(float_mul "$FIN_TUNING_LR_S1" "0.6")
    DOWNS_EN_LR_S2_BASE=$(float_mul "$DOWNS_EN_LR_S1" "0.7")

    for K in "${RESAMP_TOKENS[@]}"; do
      for CLIP_GRAD in "${CLIP_GRADS[@]}"; do

        COUNT=$((COUNT+1))
        echo "[$COUNT/$TOTAL] ============================================"

        # ==================== S1 ====================
        EX_NAME_S1="${EX_NUM}_${ds}_S1_align${align}_K${K}_down$(sci $DOWNS_EN_LR_S1)_txt$(sci $FIN_TUNING_LR_S1)_cg${CLIP_GRAD}"
        FULL_NAME_S1="${DATE_TAG}_train-${ds}_${MODEL_NAME}_${EX_NAME_S1}"   # <<< 新增：与 train.py 的 swanlab_name 完全一致

        run_job "S1" "$EX_NAME_S1" "$DEFAULT_TIMEOUT" \
          --device "$DEVICE" \
          --epochs "$S1_EPOCHS" \
          --batch_size "$BATCH" \
          --drop_prob "$DROP" \
          --downs_en_lr "$DOWNS_EN_LR_S1" \
          --fin_tuning_lr "$FIN_TUNING_LR_S1" \
          --clip_lr "$CLIP_LR_S2" \
          --weight_decay_rate "$WD" \
          --clip_grad "$CLIP_GRAD" \
          --warmup_prop "$WARMUP_S1" \
          --gradient_accumulation_steps "$GAS" \
          --min_epoch_num "$MIN_EPOCH" \
          --patience "$PATIENCE" \
          --patience_num "$PATIENCE_NUM" \
          --text_encoder "$TEXT_ENCODER" \
          --image_encoder "$IMAGE_ENCODER" \
          --use_bilstm \
          --use_image \
          --dataset_name "$ds" \
          --ex_project "$EX_PROJECT" \
          --ex_name "$EX_NAME_S1" \
          --model "$MODEL_NAME"
        rc=$?
        if [[ $rc -ne 0 ]]; then
          echo "[SKIP] S2 skipped because S1 failed: ${EX_NAME_S1}"
          continue
        fi

        # ==================== S2 多轮（r1/r2/r3...） ====================
        PREV_EX_NAME="$EX_NAME_S1"
        PREV_FULL_NAME="$FULL_NAME_S1"   # <<< 新增：S2 从 S1 的完整目录名继续

        for ridx in "${!S2_EPOCHS_ARR[@]}"; do
          R=$((ridx+1))
          EP="${S2_EPOCHS_ARR[$ridx]}"
          DEC="${S2_DECAYS[$ridx]}"
          WUP="${S2_WARMUPS[$ridx]}"

          FIN_TUNING_LR_S2_R=$(float_mul "$FIN_TUNING_LR_S2_BASE" "$DEC")
          DOWNS_EN_LR_S2_R=$(float_mul "$DOWNS_EN_LR_S2_BASE" "$DEC")

          EX_NAME_S2="${EX_NUM}_${ds}_S2r${R}_align${align}_K${K}_down$(sci $DOWNS_EN_LR_S2_R)_txt$(sci $FIN_TUNING_LR_S2_R)_cg${CLIP_GRAD}"
          FULL_NAME_S2="${DATE_TAG}_train-${ds}_${MODEL_NAME}_${EX_NAME_S2}"   # <<< 新增：本轮训练保存目录名（与 train.py 一致）

          run_job "S2r${R}" "$EX_NAME_S2" "$DEFAULT_TIMEOUT" \
            --device "$DEVICE" \
            --epochs "$EP" \
            --batch_size "$BATCH" \
            --drop_prob "$DROP" \
            --downs_en_lr "$DOWNS_EN_LR_S2_R" \
            --fin_tuning_lr "$FIN_TUNING_LR_S2_R" \
            --clip_lr "$CLIP_LR_S2" \
            --weight_decay_rate "$WD" \
            --clip_grad "$CLIP_GRAD" \
            --warmup_prop "$WUP" \
            --gradient_accumulation_steps "$GAS" \
            --min_epoch_num "$MIN_EPOCH" \
            --patience "$PATIENCE" \
            --patience_num "$PATIENCE_NUM" \
            --text_encoder "$TEXT_ENCODER" \
            --image_encoder "$IMAGE_ENCODER" \
            --dataset_name "$ds" \
            --ex_project "$EX_PROJECT" \
            --ex_name "$EX_NAME_S2" \
            --ex_nums "${EX_NUM}_S2r${R}" \
            --use_bilstm \
            --use_image \
            --resampler_tokens "$K" \
            --align_lambda "$align" \
            --vision_trainable \
            --continue_train_name "$PREV_FULL_NAME" \
            --model "$MODEL_NAME"
          rc=$?
          if [[ $rc -ne 0 ]]; then
            echo "[STOP] Later S2 rounds skipped because S2r${R} failed: ${EX_NAME_S2}"
            break
          fi

          # 下一轮从当前轮继续
          PREV_EX_NAME="$EX_NAME_S2"
          PREV_FULL_NAME="$FULL_NAME_S2"  # <<< 更新为本轮完整目录名
        done

      done
    done
  done
done

echo "=================== SUMMARY ==================="
echo "Success runs:"
cat "$SUCCESS_LIST" || true
echo
echo "Failed runs:"
cat "$FAIL_LIST" || true
echo "Logs at: $LOG_DIR"
/usr/bin/shutdown -h now
