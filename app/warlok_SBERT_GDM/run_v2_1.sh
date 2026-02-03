python3 train_v2_1_contrastive.py \
  --docs \
  "docs/Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt" \
  "docs/Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt" \
  "docs/A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt" \
  "docs/Diabetes_mellitus_and_pregnancy.txt" \
  --out_dir models \
  --name gdm_sbert_v2_1 \
  --epochs 18 \
  --batch 64 \
  --train_pairs 20000 \
  --train_embed_rows 2500 \
  --print_steps
