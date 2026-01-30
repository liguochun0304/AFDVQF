STAMP=$(date +%F_%H%M%S)
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_${STAMP}.log"

python ../test.py --save_name "2026-01-30_train-twitter2015_mqspn_original_mqspn_original_crf" --device cuda:0 |& tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
/usr/bin/shutdown -h now
exit $rc
#
#
#2025-08-14_train-twitter2015_MNER_B1_twitter2015_roberta_clip_align0/
#2025-08-14_train-twitter2015_MNER_B2_twitter2015_roberta_clip_align0.05/
#2025-08-14_train-twitter2015_MNER_B3_twitter2015_roberta_clip_align0.1/
#2025-08-14_train-twitter2015_MNER_B4_twitter2015_roberta_clip_align0.2/
#2025-08-14_train-twitter2017_MNER_B1_twitter2017_roberta_clip_align0/
#2025-08-14_train-twitter2017_MNER_B2_twitter2017_roberta_clip_align0.05/
#2025-08-14_train-twitter2017_MNER_B3_twitter2017_roberta_clip_align0.1/
#2025-08-14_train-twitter2017_MNER_B4_twitter2017_roberta_clip_align0.2/
#
