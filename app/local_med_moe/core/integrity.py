# core/integrity.py
def integrity_decision(cfg, router_conf: float, energies, gates: dict, entailment=None):
    flags = []

    # required gates (if present)
    for k in ["intent_ok", "role_ok", "schema_ok", "failure_ok", "hormone_ok"]:
        if k in gates and not gates.get(k, True):
            flags.append(k.replace("_ok", "") + "_mismatch")

    if router_conf < cfg.router_conf_min:
        flags.append("low_router_conf")

    if energies:
        bestE = min(energies)
        if bestE > cfg.energy_good_max:
            flags.append("high_energy")
    else:
        flags.append("no_candidates")

    if entailment is not None and not entailment.get("passed", False):
        flags.append(f"classification_not_entitled:{entailment.get('support_type','none')}")

    decision = "ALLOW" if not flags else "ABSTAIN"
    return {"decision": decision, "flags": flags}

# # core/integrity.py
# def integrity_decision(cfg, router_conf: float, energies, gates: dict, entailment=None):
#     """
#     gates: dict with keys:
#       intent_ok, role_ok, schema_ok
#     """
#     flags = []
#     if not gates.get("intent_ok", True):
#         flags.append("intent_mismatch")
#     if not gates.get("role_ok", True):
#         flags.append("role_mismatch")
#     if not gates.get("schema_ok", True):
#         flags.append("schema_missing")
#
#     if router_conf < cfg.router_conf_min:
#         flags.append("low_router_conf")
#
#     if energies:
#         bestE = min(energies)
#         if bestE > cfg.energy_good_max:
#             flags.append("high_energy")
#     else:
#         flags.append("no_candidates")
#
#     if entailment is not None:
#         if not entailment.get("passed", False):
#             flags.append(f"classification_not_entitled:{entailment.get('support_type','none')}")
#
#     decision = "ALLOW" if not flags else "ABSTAIN"
#     return {"decision": decision, "flags": flags}
