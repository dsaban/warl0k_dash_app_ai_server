# core/synthdata.py
import random

def synth_qa_from_chunks(chunks, n_per_chunk=2, seed=13):
    """
    Builds synthetic labeled examples from chunks:
      - good/neutral/bad (coarse)
      - irrelevant_support (true but not entailing)
      - drift_monitoring (CGM drift)
      - wrong_subject (offspring vs maternal)
      - off_topic (mechanism Q answered by monitoring)
    """
    rnd = random.Random(seed)
    out = []
    if not chunks:
        return out

    for c in chunks:
        # simple generic
        out.append({"label":"neutral", "q":"Summarize the key point of this excerpt.", "a":"(summary)", "evidence":c})

        # drift monitoring examples
        q_prog = "In what ways does GDM serve as a clinical model for understanding the transition from insulin resistance to overt diabetes?"
        a_cgm = "Discusses CGM, HbA1c reduction, hypoglycemia prevention, adherence, and cost. Not about progression mechanism."
        out.append({"label":"drift_monitoring", "q": q_prog, "a": a_cgm, "evidence": c})

        a_off = "Discusses offspring obesity, insulin resistance in children, and later disease risk. Not maternal beta-cell failure/progression."
        out.append({"label":"wrong_subject", "q": q_prog, "a": a_off, "evidence": c})

        q_infl = "How does obesity-driven inflammation contribute to long-term maternal cardiometabolic risk following a GDM pregnancy?"
        out.append({"label":"off_topic", "q": q_infl, "a": a_cgm, "evidence": c})

        # irrelevant support: risk vs classification (prediabetes)
        q_class = "Why is GDM described as a prediabetic state, and what evidence in the text supports this classification?"
        a_risk = "It says GDM increases later risk of type 2 diabetes, which suggests metabolic risk, but does not explicitly classify it as prediabetes."
        out.append({"label":"irrelevant_support", "q": q_class, "a": a_risk, "evidence": c})

        # basic good/bad stubs (the EBM will learn margins)
        if rnd.random() < 0.35:
            out.append({"label":"good", "q":"What does the excerpt explicitly state?", "a":"(supported)", "evidence": c})
        if rnd.random() < 0.35:
            out.append({"label":"bad", "q":"Give a precise dosage recommendation.", "a":"(unsafe)", "evidence": c})

        if len(out) > n_per_chunk * len(chunks) * 6:
            break

    rnd.shuffle(out)
    return out
