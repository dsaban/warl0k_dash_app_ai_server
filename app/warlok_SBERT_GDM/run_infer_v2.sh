python3 infer_demo_v2A.py \
  --models_dir models \
  --name gdm_sbert_v2_1 \
  --docs \
  "docs/Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt" \
  "docs/Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt" \
  "docs/A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt" \
  "docs/Diabetes_mellitus_and_pregnancy.txt" \
  --demo tests/demo_questions.jsonl \
  -k 10
