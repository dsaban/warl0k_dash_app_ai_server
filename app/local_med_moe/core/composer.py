# core/composer.py

def compose_from_schema(qtype_name: str, schema: dict) -> str:
    filled = schema.get("filled", [])
    by_slot = {x["slot"]: x["sentence"] for x in filled}

    def get(slot):
        return by_slot.get(slot, "").strip()

    if qtype_name == "placental_hormones_pathologic_gdm":
        s1 = get("Placental hormones → insulin resistance")
        s2 = get("Physiological purpose")
        s3 = get("Beta-cell compensation")
        s4 = get("Pathological transition in GDM")

        lines = []
        if s1:
            lines.append(f"Placental hormones increase maternal insulin resistance during pregnancy. ({s1})")
        if s2:
            lines.append(f"This physiological insulin resistance helps prioritize glucose/nutrients for the fetus. ({s2})")
        if s3:
            lines.append(f"Normally, beta-cells compensate by increasing insulin secretion to maintain normoglycemia. ({s3})")
        if s4:
            lines.append(f"In GDM, this compensation is insufficient, so insulin resistance becomes pathological and hyperglycemia develops. ({s4})")

        return "\n".join(lines) if lines else "(No supported synthesis found.)"

    if qtype_name == "maternal_beta_cell_failure":
        s1 = get("Pregnancy-induced insulin resistance")
        s2 = get("Increased insulin demand")
        s3 = get("Beta-cell compensation")
        s4 = get("Failure in susceptible women")
        s5 = get("Result: hyperglycemia / GDM")
        s6 = get("Susceptibility factors (if present)")

        lines = []
        if s1:
            lines.append(f"Pregnancy increases insulin resistance, raising insulin requirements. ({s1})")
        if s2:
            lines.append(f"This elevates insulin demand beyond baseline. ({s2})")
        if s3:
            lines.append(f"Normally, pancreatic beta-cells compensate by increasing insulin secretion to maintain euglycemia. ({s3})")
        if s4:
            lines.append(f"In some women this compensation is inadequate due to beta-cell dysfunction/limited reserve, so glucose control breaks down. ({s4})")
        if s5:
            lines.append(f"The result is hyperglycemia during pregnancy, meeting criteria for GDM. ({s5})")
        if s6:
            lines.append(f"Susceptibility factors are mentioned only where supported by the text. ({s6})")

        return "\n".join(lines) if lines else "(No supported synthesis found.)"

    # Default: list filled slots
    out = [f"QType: {qtype_name}"]
    for it in filled:
        out.append(f"- {it['slot']}: {it['sentence']}")
    if schema.get("missing"):
        out.append("\nNot supported by provided text:")
        for m in schema["missing"]:
            out.append(f"- {m['slot']}: {m['guidance']}")
    return "\n".join(out)

# # core/composer.py
# from .utils import normalize_text
#
# def compose_from_schema(qtype_name: str, schema: dict) -> str:
#     """
#     Deterministic synthesis: uses ONLY filled schema sentences.
#     Produces a causal chain instead of a list.
#     """
#     filled = schema.get("filled", [])
#     by_slot = {x["slot"]: x["sentence"] for x in filled}
#
#     def get(slot):
#         return by_slot.get(slot, "").strip()
#
#     if qtype_name == "maternal_beta_cell_failure":
#         s1 = get("Pregnancy-induced insulin resistance")
#         s2 = get("Increased insulin demand")
#         s3 = get("Beta-cell compensation")
#         s4 = get("Failure in susceptible women")
#         s5 = get("Result: hyperglycemia / GDM")
#         s6 = get("Susceptibility factors (if present)")
#
#         lines = []
#         if s1: lines.append(f"Pregnancy increases insulin resistance, raising insulin requirements. ({s1})")
#         if s2: lines.append(f"This elevates insulin demand beyond baseline. ({s2})")
#         if s3: lines.append(f"Normally, pancreatic beta-cells compensate by increasing insulin secretion to maintain euglycemia. ({s3})")
#         if s4: lines.append(f"In some women this compensation is inadequate due to beta-cell dysfunction/limited reserve, so glucose control breaks down. ({s4})")
#         if s5: lines.append(f"The result is hyperglycemia during pregnancy, meeting criteria for GDM. ({s5})")
#         if s6: lines.append(f"Susceptibility factors are noted where supported by the text. ({s6})")
#
#         return "\n".join(lines) if lines else "(No supported synthesis found.)"
#
#     # Default: list filled slots
#     out = [f"QType: {qtype_name}"]
#     for it in filled:
#         out.append(f"- {it['slot']}: {it['sentence']}")
#     if schema.get("missing"):
#         out.append("\nNot supported by provided text:")
#         for m in schema["missing"]:
#             out.append(f"- {m['slot']}: {m['guidance']}")
#     return "\n".join(out)
