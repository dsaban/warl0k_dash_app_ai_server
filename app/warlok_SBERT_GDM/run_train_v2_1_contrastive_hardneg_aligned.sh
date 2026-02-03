python3 train_v2_1_contrastive_hardneg_aligned.py \
  --docs \
  "docs/Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt" \
  "docs/Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt" \
  "docs/A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt" \
  "docs/Diabetes_mellitus_and_pregnancy.txt" \
  --out_dir models \
  --name gdm_sbert_v2_1_hardneg_aligned \
  --print_steps

python3 infer_demo_v2A.py \
  --models_dir models \
  --name gdm_sbert_v2_1_hardneg_aligned \
  --docs \
  "docs/Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt" \
  "docs/Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt" \
  "docs/A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt" \
  "docs/Diabetes_mellitus_and_pregnancy.txt" \
  --demo tests/demo_questions.jsonl \
  -k 10

