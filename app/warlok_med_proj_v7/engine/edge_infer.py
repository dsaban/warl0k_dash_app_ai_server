import re
from typing import List

INFER = {
    "CROSSES_PLACENTA": re.compile(
        r'glucose.*(cross|transfer|pass|diffus).*placenta', re.I
    ),

    "STIMULATES_INSULIN": re.compile(
        r'(fetal|foetal).*insulin', re.I
    ),

    "PROMOTES_GROWTH": re.compile(
        r'(fetal|foetal).*insulin.*(growth|macrosomia|lga)', re.I
    ),

    # FIXED: postpartum OR after delivery + flexible timing
    "POSTPARTUM_OGTT": re.compile(
        r'(postpartum|after delivery|after giving birth).{0,120}?'
        r'(75\s*g|75g)?.{0,80}?'
        r'(ogtt|oral glucose tolerance test).{0,120}?'
        r'(4\s*-\s*12\s*weeks|6\s*weeks|6\s*weeks?\s*or\s*more)',
        re.I
    ),

    "FOLLOWUP_GAP": re.compile(
        r'(postpartum|after delivery).*(follow|attendance|testing).*(low|poor|missed|gap)',
        re.I
    ),

    "MISSED_DETECTION": re.compile(
        r'(missed|delayed).*(diagnos|detect)', re.I
    ),

    # NEW: transient proof
    "RESOLVES_AFTER_DELIVERY": re.compile(
        r'(gdm|gestational diabetes).{0,120}?(resolves|resolve).{0,120}?'
        r'(after delivery|postpartum|after giving birth)',
        re.I
    ),
}

def infer_edges(text: str) -> List[str]:
    return [e for e, pat in INFER.items() if pat.search(text)]

# import re
# from typing import List
#
# INFER = {
#     "CROSSES_PLACENTA": re.compile(r'glucose.*(cross|transfer|pass|diffus).*placenta', re.I),
#     "STIMULATES_INSULIN": re.compile(r'(fetal|foetal).*insulin', re.I),
#     "PROMOTES_GROWTH": re.compile(r'(fetal|foetal).*insulin.*(growth|macrosomia|lga)', re.I),
#     "POSTPARTUM_OGTT": re.compile(r'postpartum.*(ogtt|oral glucose tolerance).*weeks', re.I),
#     "FOLLOWUP_GAP": re.compile(r'postpartum.*(follow|attendance|testing).*(low|poor|missed|gap)', re.I),
#     "MISSED_DETECTION": re.compile(r'(missed|delayed).*(diagnos|detect)', re.I),
# }
#
# def infer_edges(text: str) -> List[str]:
#     return [e for e, pat in INFER.items() if pat.search(text)]
