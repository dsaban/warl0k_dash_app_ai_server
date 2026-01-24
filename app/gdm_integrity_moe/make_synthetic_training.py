import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tq = [
        {"question":"Explain how placental hormones contribute to insulin resistance during pregnancy and why this becomes pathological in GDM.",
         "qtype":"mechanism_hormone_to_gdm","target_expert":"mechanism"},
        {"question":"Why is ethnicity considered an independent risk factor for GDM and how is this linked to background population rates of type 2 diabetes?",
         "qtype":"risk_ethnicity_t2d_link","target_expert":"risk_ethnicity"},
    ]
    with open(os.path.join(args.out_dir, "train_questions.jsonl"), "w", encoding="utf-8") as f:
        for r in tq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tp = [
        {"question":tq[0]["question"], "label":"good"},
        {"question":tq[1]["question"], "label":"good"},
        {"question":"How does continuous glucose monitoring reduce HbA1c in GDM?", "label":"neutral"},
    ]
    with open(os.path.join(args.out_dir, "train_pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in tp:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Wrote synthetic training files to", args.out_dir)

if __name__ == "__main__":
    main()
