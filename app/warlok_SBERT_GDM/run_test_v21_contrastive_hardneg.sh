python3 train_v2_1_contrastive_hardneg.py \
  --docs \
  "docs/Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt" \
  "docs/Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt" \
  "docs/A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt" \
  "docs/Diabetes_mellitus_and_pregnancy.txt" \
  --out_dir models \
  --name gdm_sbert_v2_1_hardneg_tuned \
  --epochs 28 \
  --batch 64 \
  --train_pairs 45000 \
  --train_embed_rows 8000 \
  --tau 0.14 \
  --lr 1.5e-3 \
  --lr_pool 8e-4 \
  --lr_gate 4e-4 \
  --lr_emb 9e-4 \
  --p_hard 0.85 \
  --hard_topk 60 \
  --mine_refresh_steps 60 \
  --bank_max_sents 3500 \
  --bank_meta_stride 20 \
  --print_steps \
  --print_every 25

python3 infer_demo_v2A.py \
  --models_dir models \
  --name gdm_sbert_v2_1_hardneg_tuned \
  --docs \
  "docs/Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt" \
  "docs/Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt" \
  "docs/A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt" \
  "docs/Diabetes_mellitus_and_pregnancy.txt" \
  --demo tests/demo_questions.jsonl \
  -k 10
