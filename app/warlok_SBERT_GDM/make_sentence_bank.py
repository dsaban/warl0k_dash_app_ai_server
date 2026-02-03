import re
from pathlib import Path
import os

#  set docs path to files
root = Path(__file__).parent
print(f"Current script directory: {root}")

path_to_docs = f"{root / 'docs'}"
print(f"Current working directory: {path_to_docs}")

docs = [
	"Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt",
	"Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt",
	"A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt",
	"Diabetes_mellitus_and_pregnancy.txt",
]

def split_sentences(text):
	text = re.sub(r"\s+", " ", text)
	return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 60 < len(s) < 350]

sentences = []
for doc in docs:
	# call read_text with errors="ignore"
	text = Path(path_to_docs).joinpath(doc).read_text(errors="ignore")
	sentences.extend(split_sentences(text))
	
	# text = path_to_docs.joinpath(doc).read_text(errors="ignore")
	# sentences.extend(split_sentences(text))

# remove duplicates
sentences = list(dict.fromkeys(sentences))

Path("sentence_bank.txt").write_text("\n".join(sentences))
print(f"Saved {len(sentences)} sentences to sentence_bank.txt")
