from .utils import norm

def detect_qtype(q: str) -> str:
    s = norm(q)

    # ethnicity + gdm + population/type2 link
    if any(k in s for k in ["ethnicity","race","ethnic","racial"]) and ("gdm" in s or "gestational" in s) and any(k in s for k in ["type 2","t2d","prevalence","population","background"]):
        return "risk_ethnicity_t2d_link"

    # placental hormones + insulin resistance + pathological/GDM
    if any(k in s for k in ["placental","hpl","progesterone","cortisol","growth hormone","estrogen","prolactin"]) and ("insulin resistance" in s) and any(k in s for k in ["gdm","gestational","pathological","pathology"]):
        return "mechanism_hormone_to_gdm"

    return "generic"
