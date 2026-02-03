#python3 infer_retriever_only_v3.py \
#  --models_dir models \
#  --name gdm_retriever_only_v3 \
#  --docs \
#  "docs/Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt" \
#  "docs/Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt" \
#  "docs/A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt" \
#  "docs/Diabetes_mellitus_and_pregnancy.txt" \
#  --evidence_only \
#  -k 10 \
#  --query "When is screening for GDM performed and what test is used?"

python3 infer_retriever_only_v3.py \
  --models_dir models \
  --name gdm_retriever_only_v3 \
  --docs \
  "docs/Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt" \
  "docs/Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt" \
  "docs/A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt" \
  "docs/Diabetes_mellitus_and_pregnancy.txt" \
  --evidence_only \
  -k 10 \
  --query_file tests/demo_questions.jsonl
