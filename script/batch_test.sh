#!/usr/bin/env bash
# 仅按给定列表测试指定保存目录，并汇总指标到 CSV
# 用法：bash run_eval_list.sh

set -u

PY=python3
TEST=../test.py
SAVE_ROOT="/workspace/save_models"   # 如路径不同，改这里
DEVICE="${DEVICE:-cuda:0}"           # 可用 DEVICE=cuda:1 覆盖

# ====== 只测试这份名单（按需修改）======
save_names=(
  "2025-08-15_train-twitter2015_MNER_B1_twitter2015_S1_align0_K4_down1e-04_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B1_twitter2015_S1_align0_K8_down1e-04_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B1_twitter2015_S1_roberta_clip_align0_K4_down1e-04_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B1_twitter2015_S1_roberta_clip_align0_K8_down1e-04_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B1_twitter2015_S2_align0_K4_down7e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B1_twitter2015_S2_roberta_clip_align0_K4_down7e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B1_twitter2015_S2r1_align0_K4_down7e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B1_twitter2015_S2r1_align0_K8_down7e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B2_twitter2015_S1_align0.05_K4_down9e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B2_twitter2015_S1_align0.05_K8_down9e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B2_twitter2015_S2r1_align0.05_K4_down6e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B2_twitter2015_S2r1_align0.05_K8_down6e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B3_twitter2015_S1_align0.1_K4_down9e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B3_twitter2015_S1_align0.1_K8_down9e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B3_twitter2015_S2r1_align0.1_K4_down6e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B3_twitter2015_S2r1_align0.1_K8_down6e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B4_twitter2015_S1_align0.2_K4_down8e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B4_twitter2015_S1_align0.2_K8_down8e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B4_twitter2015_S2r1_align0.2_K4_down6e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B4_twitter2015_S2r1_align0.2_K8_down6e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B5_twitter2015_S1_align0.4_K4_down7e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B5_twitter2015_S1_align0.4_K8_down7e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B5_twitter2015_S2r1_align0.4_K4_down5e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B5_twitter2015_S2r1_align0.4_K8_down5e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B6_twitter2015_S1_align0.8_K4_down6e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B6_twitter2015_S1_align0.8_K8_down6e-05_txt4e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B6_twitter2015_S2r1_align0.8_K4_down4e-05_txt2e-05_cg1.0"
  "2025-08-15_train-twitter2015_MNER_B6_twitter2015_S2r1_align0.8_K8_down4e-05_txt2e-05_cg1.0"

  "2025-08-15_train-twitter2017_MNER_B1_twitter2017_S1_align0_K4_down1e-04_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B1_twitter2017_S1_align0_K8_down1e-04_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B1_twitter2017_S2r1_align0_K4_down8e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B1_twitter2017_S2r1_align0_K8_down8e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B2_twitter2017_S1_align0.05_K4_down1e-04_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B2_twitter2017_S1_align0.05_K8_down1e-04_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B2_twitter2017_S2r1_align0.05_K4_down8e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B2_twitter2017_S2r1_align0.05_K8_down8e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B3_twitter2017_S1_align0.1_K4_down1e-04_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B3_twitter2017_S1_align0.1_K8_down1e-04_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B3_twitter2017_S2r1_align0.1_K4_down8e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B3_twitter2017_S2r1_align0.1_K8_down8e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B4_twitter2017_S1_align0.2_K4_down1e-04_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B4_twitter2017_S1_align0.2_K8_down1e-04_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B4_twitter2017_S2r1_align0.2_K4_down7e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B4_twitter2017_S2r1_align0.2_K8_down7e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B5_twitter2017_S1_align0.4_K4_down8e-05_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B5_twitter2017_S1_align0.4_K8_down8e-05_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B5_twitter2017_S2r1_align0.4_K4_down6e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B5_twitter2017_S2r1_align0.4_K8_down6e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B6_twitter2017_S1_align0.8_K4_down7e-05_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B6_twitter2017_S1_align0.8_K8_down7e-05_txt5e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B6_twitter2017_S2r1_align0.8_K4_down5e-05_txt3e-05_cg1.0"
  "2025-08-15_train-twitter2017_MNER_B6_twitter2017_S2r1_align0.8_K8_down5e-05_txt3e-05_cg1.0"
)


STAMP=$(date +%F_%H%M%S)
LOG_DIR="logs_test_${STAMP}"
mkdir -p "$LOG_DIR"

SUMMARY_CSV="$LOG_DIR/summary.csv"
echo "run_name,acc,precision,recall,f1,per_f1,loc_f1,org_f1,misc_f1,log_file" > "$SUMMARY_CSV"

# 解析 test.py 输出（适配你现有格式）
parse_metrics_from_log() {
  local log="$1"
  local acc p r f1 per loc org misc

  acc=$(grep -Eo '\[Overall\].*Acc=[0-9. ]+' "$log" | tail -n1 | sed -E 's/.*Acc=([0-9.]+).*/\1/')
  p=$(grep -Eo '\[Overall\].*P=[0-9. ]+' "$log" | tail -n1 | sed -E 's/.*P=([0-9.]+).*/\1/')
  r=$(grep -Eo '\[Overall\].*R=[0-9. ]+' "$log" | tail -n1 | sed -E 's/.*R=([0-9.]+).*/\1/')
  f1=$(grep -Eo '\[Overall\].*F1=[0-9. ]+' "$log" | tail -n1 | sed -E 's/.*F1=([0-9.]+).*/\1/')

  per=$(grep -E '^\[PER\]' "$log"  | tail -n1 | sed -E 's/.*F1=([0-9.]+).*/\1/')
  loc=$(grep -E '^\[LOC\]' "$log"  | tail -n1 | sed -E 's/.*F1=([0-9.]+).*/\1/')
  org=$(grep -E '^\[ORG\]' "$log"  | tail -n1 | sed -E 's/.*F1=([0-9.]+).*/\1/')
  misc=$(grep -E '^\[MISC\]' "$log" | tail -n1 | sed -E 's/.*F1=([0-9.]+).*/\1/')

  echo "${acc:-NA},${p:-NA},${r:-NA},${f1:-NA},${per:-NA},${loc:-NA},${org:-NA},${misc:-NA}"
}

for name in "${save_names[@]}"; do
  SAVE_DIR="$SAVE_ROOT/$name"
  MODEL_PT="$SAVE_DIR/model.pt"

  echo "开始运行: $name"
  if [[ ! -f "$MODEL_PT" ]]; then
    echo "[SKIP] 缺少模型权重：$MODEL_PT"
    echo "$name,NA,NA,NA,NA,NA,NA,NA,NA,missing_model" >> "$SUMMARY_CSV"
    echo "完成: $name"
    echo "-----------------------"
    continue
  fi

  log_file="$LOG_DIR/${name//\//_}.log"  # 防止名字里含斜杠
  $PY "$TEST" --save_name "$name" --device "$DEVICE" |& tee "$log_file"
  rc=${PIPESTATUS[0]}
  echo "完成: $name (rc=$rc)"
  echo "-----------------------"

  metrics=$(parse_metrics_from_log "$log_file")
  echo "$name,${metrics},$log_file" >> "$SUMMARY_CSV"
done

echo
echo "======== 全部完成 ========"
echo "汇总表：$SUMMARY_CSV"
echo "日志目录：$LOG_DIR"



2025-08-15_train-twitter2015_MNER_B1_twitter2015_S1_align0_K4_down1e-04_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B1_twitter2015_S1_align0_K8_down1e-04_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B1_twitter2015_S1_roberta_clip_align0_K4_down1e-04_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B1_twitter2015_S1_roberta_clip_align0_K8_down1e-04_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B1_twitter2015_S2_align0_K4_down7e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B1_twitter2015_S2_roberta_clip_align0_K4_down7e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B1_twitter2015_S2r1_align0_K4_down7e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B1_twitter2015_S2r1_align0_K8_down7e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B2_twitter2015_S1_align0.05_K4_down9e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B2_twitter2015_S1_align0.05_K8_down9e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B2_twitter2015_S2r1_align0.05_K4_down6e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B2_twitter2015_S2r1_align0.05_K8_down6e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B3_twitter2015_S1_align0.1_K4_down9e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B3_twitter2015_S1_align0.1_K8_down9e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B3_twitter2015_S2r1_align0.1_K4_down6e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B3_twitter2015_S2r1_align0.1_K8_down6e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B4_twitter2015_S1_align0.2_K4_down8e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B4_twitter2015_S1_align0.2_K8_down8e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B4_twitter2015_S2r1_align0.2_K4_down6e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B4_twitter2015_S2r1_align0.2_K8_down6e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B5_twitter2015_S1_align0.4_K4_down7e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B5_twitter2015_S1_align0.4_K8_down7e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B5_twitter2015_S2r1_align0.4_K4_down5e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B5_twitter2015_S2r1_align0.4_K8_down5e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B6_twitter2015_S1_align0.8_K4_down6e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B6_twitter2015_S1_align0.8_K8_down6e-05_txt4e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B6_twitter2015_S2r1_align0.8_K4_down4e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2015_MNER_B6_twitter2015_S2r1_align0.8_K8_down4e-05_txt2e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B1_twitter2017_S1_align0_K4_down1e-04_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B1_twitter2017_S1_align0_K8_down1e-04_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B1_twitter2017_S2r1_align0_K4_down8e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B1_twitter2017_S2r1_align0_K8_down8e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B2_twitter2017_S1_align0.05_K4_down1e-04_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B2_twitter2017_S1_align0.05_K8_down1e-04_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B2_twitter2017_S2r1_align0.05_K4_down8e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B2_twitter2017_S2r1_align0.05_K8_down8e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B3_twitter2017_S1_align0.1_K4_down1e-04_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B3_twitter2017_S1_align0.1_K8_down1e-04_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B3_twitter2017_S2r1_align0.1_K4_down8e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B3_twitter2017_S2r1_align0.1_K8_down8e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B4_twitter2017_S1_align0.2_K4_down1e-04_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B4_twitter2017_S1_align0.2_K8_down1e-04_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B4_twitter2017_S2r1_align0.2_K4_down7e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B4_twitter2017_S2r1_align0.2_K8_down7e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B5_twitter2017_S1_align0.4_K4_down8e-05_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B5_twitter2017_S1_align0.4_K8_down8e-05_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B5_twitter2017_S2r1_align0.4_K4_down6e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B5_twitter2017_S2r1_align0.4_K8_down6e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B6_twitter2017_S1_align0.8_K4_down7e-05_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B6_twitter2017_S1_align0.8_K8_down7e-05_txt5e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B6_twitter2017_S2r1_align0.8_K4_down5e-05_txt3e-05_cg1.0/
2025-08-15_train-twitter2017_MNER_B6_twitter2017_S2r1_align0.8_K8_down5e-05_txt3e-05_cg1.0/
