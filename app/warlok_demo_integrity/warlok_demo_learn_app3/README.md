# WARL0K GDM Integrity Demo 1 (Streamlit + NumPy-only core)

This demo wraps the evidence-locked Q&A prototype into a simple Streamlit app.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What you can do
- Ask a question → get an evidence-locked, concise answer (max sentences)
- Paste a free answer (e.g., ChatGPT) → get an integrity label + support metrics
- Batch benchmark the provided 100-question eval set and optionally upload your own `free_answers.jsonl`

## Files
- app.py: Streamlit UI
- core.py: retrieval + composition + scoring (NumPy)
- claims_200.jsonl: claim store
- eval_set_100.jsonl / eval_set_100.csv: evaluation set
- compare_scoring.py: CLI scorer (optional)

## Question bank
Use the **Question bank (Eval 100)** tab to pick any eval question, paste a GPT answer, score it, and export a `free_answers.jsonl` file for batch comparison.

## Learned evaluator
The app trains a small NumPy-only softmax classifier from the synthetic eval variants and uses it to label answers with probabilities.
