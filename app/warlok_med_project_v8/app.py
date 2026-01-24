import streamlit as st
from pathlib import Path

from engine.pipeline import ensure_dirs, save_uploaded_docs, build_all
from engine.domain_pack import DomainPack
from engine.qa import answer


def list_packs(packs_dir: Path):
    if not packs_dir.exists():
        return []
    packs = []
    for p in packs_dir.iterdir():
        if p.is_dir() and (p / "manifest.json").exists():
            packs.append(p)
    return sorted(packs, key=lambda x: x.name)


def main():
    st.set_page_config(page_title="GDM ChainGraph v8", layout="wide")
    st.title("GDM ChainGraph v8 — Domain Packs")

    ROOT = Path(".").resolve()
    paths = ensure_dirs(ROOT)

    DOCS_DIR = paths["docs_dir"]
    INDEX_DIR = paths["index_dir"]
    EVAL_DIR = paths["eval_dir"]

    PACKS_DIR = ROOT / "domain_packs"
    packs = list_packs(PACKS_DIR)
    if not packs:
        st.error("No packs found. Create: domain_packs/gdm_v1/manifest.json")
        st.stop()

    st.sidebar.header("Domain Pack")
    pack_names = [p.name for p in packs]
    sel = st.sidebar.selectbox("Select pack", pack_names, index=0)
    PACK_DIR = PACKS_DIR / sel

    try:
        dp = DomainPack.load(PACK_DIR)
        st.sidebar.success(f"{dp.manifest.get('name')} v{dp.manifest.get('version')}")
        st.sidebar.caption(str(PACK_DIR))
    except Exception as e:
        st.sidebar.error(f"Pack load error: {e}")
        st.stop()

    tab_upload, tab_build, tab_ask, tab_pack = st.tabs(
        ["Upload docs", "Build index", "Ask", "Pack viewer"]
    )

    with tab_upload:
        st.subheader("1) Upload your .txt documents")
        st.write(f"Saved into: `{DOCS_DIR}`")

        uploaded = st.file_uploader("Upload .txt docs", type=["txt"], accept_multiple_files=True)
        overwrite = st.checkbox("Overwrite if same filename exists", value=True)

        if st.button("Save uploaded docs", type="primary"):
            res = save_uploaded_docs(DOCS_DIR, uploaded, overwrite=overwrite)
            st.success(f"Saved: {len(res['saved'])}, Skipped: {len(res['skipped'])}")
            if res["saved"]:
                st.write("Saved:")
                for p in res["saved"]:
                    st.code(p)
            if res["skipped"]:
                st.write("Skipped:")
                st.json(res["skipped"])

        st.divider()
        files = sorted([p.name for p in DOCS_DIR.glob("*.txt")])
        st.write("Currently in docs/:")
        st.dataframe([{"file": f} for f in files] if files else [{"file": "(none)"}])

    with tab_build:
        st.subheader("2) Build / Rebuild index & ontology")
        st.write(f"Using pack: `{PACK_DIR}`")

        run_eval = st.checkbox("Also generate questions + self-eval", value=True)
        eval_limit = st.slider("Eval question limit", min_value=50, max_value=3000, value=600, step=50)

        if st.button("Build now", type="primary"):
            br = build_all(ROOT, PACK_DIR, run_eval=run_eval, eval_limit=eval_limit)
            if br.ok:
                st.success(br.message)
                st.json(br.stats)
            else:
                st.error(br.message)

        st.divider()
        st.write("Index status:")
        st.write(f"- ontology.json exists: `{(INDEX_DIR / 'ontology.json').exists()}`")
        st.write(f"- claims.jsonl exists: `{(INDEX_DIR / 'claims.jsonl').exists()}`")

    with tab_ask:
        st.subheader("3) Ask")
        if not (INDEX_DIR / "claims.jsonl").exists():
            st.warning("Index not built yet. Go to **Build index** and click **Build now**.")
        else:
            q = st.text_area(
                "Question",
                height=120,
                value="Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia."
            )
            if st.button("Answer", type="primary"):
                res = answer(INDEX_DIR, q, dp)
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown(res.markdown)
                with c2:
                    st.subheader("Debug")
                    st.json(res.debug)

    with tab_pack:
        st.subheader("Domain Pack Viewer")
        st.write(f"Pack dir: `{PACK_DIR}`")
        st.json(dp.manifest)

        with st.expander("Frames"):
            st.json(dp.frames_raw)

        with st.expander("Routing rules"):
            st.json(dp.routing_rules)

        with st.expander("Edge patterns (raw)"):
            st.json(dp.edge_patterns_raw)

        with st.expander("Infer rules (raw)"):
            st.json(dp.infer_rules_raw)

        with st.expander("Question templates"):
            st.json(dp.question_templates)

        with st.expander("Junk filters"):
            st.json(dp.junk_filters)

        with st.expander("Scoring"):
            st.json(dp.scoring)


if __name__ == "__main__":
    main()

# import streamlit as st
# from pathlib import Path
#
# from engine.pipeline import ensure_dirs, save_uploaded_docs, build_all
# from engine.domain_pack import DomainPack
# from engine.qa import answer
#
#
# def list_packs(packs_dir: Path):
#     if not packs_dir.exists():
#         return []
#     packs = []
#     for p in packs_dir.iterdir():
#         if p.is_dir() and (p / "manifest.json").exists():
#             packs.append(p)
#     return sorted(packs, key=lambda x: x.name)
#
#
# def main():
#     st.set_page_config(page_title="GDM ChainGraph — Domain Packs", layout="wide")
#     st.title("GDM ChainGraph — Domain Packs (external frames/patterns/routing)")
#
#     ROOT = Path(".").resolve()
#     paths = ensure_dirs(ROOT)
#     DOCS_DIR = paths["docs_dir"]
#     INDEX_DIR = paths["index_dir"]
#     EVAL_DIR = paths["eval_dir"]
#
#     PACKS_DIR = ROOT / "domain_packs"
#     packs = list_packs(PACKS_DIR)
#
#     if not packs:
#         st.error(
#             "No domain packs found.\n\n"
#             "Create: `domain_packs/gdm_v1/manifest.json` (and the referenced JSON files)."
#         )
#         st.stop()
#
#     # Sidebar: choose pack
#     st.sidebar.header("Domain Pack")
#     pack_names = [p.name for p in packs]
#     sel = st.sidebar.selectbox("Select pack", pack_names, index=0)
#     PACK_DIR = PACKS_DIR / sel
#
#     # Load pack safely (no crash loops)
#     try:
#         dp = DomainPack.load(PACK_DIR)
#         st.sidebar.success(f"{dp.manifest.get('name')} v{dp.manifest.get('version')}")
#         st.sidebar.caption(str(PACK_DIR))
#     except Exception as e:
#         st.sidebar.error(f"Pack load error: {e}")
#         st.stop()
#
#     tab_upload, tab_build, tab_ask, tab_pack = st.tabs(
#         ["Upload docs", "Build index", "Ask", "Pack viewer"]
#     )
#
#     with tab_upload:
#         st.subheader("1) Upload your .txt documents")
#         st.write(f"Docs will be saved into: `{DOCS_DIR}`")
#
#         uploaded = st.file_uploader("Upload .txt docs", type=["txt"], accept_multiple_files=True)
#         overwrite = st.checkbox("Overwrite if same filename exists", value=True)
#
#         if st.button("Save uploaded docs", type="primary"):
#             res = save_uploaded_docs(DOCS_DIR, uploaded, overwrite=overwrite)
#             st.success(f"Saved: {len(res['saved'])}, Skipped: {len(res['skipped'])}")
#             if res["saved"]:
#                 st.write("Saved files:")
#                 for p in res["saved"]:
#                     st.code(p)
#             if res["skipped"]:
#                 st.write("Skipped:")
#                 st.json(res["skipped"])
#
#         st.divider()
#         st.write("Currently in `docs/`:")
#         files = sorted([p.name for p in DOCS_DIR.glob("*.txt")])
#         st.dataframe([{"file": f} for f in files] if files else [{"file": "(none)"}])
#
#     with tab_build:
#         st.subheader("2) Build / Rebuild index & ontology (using selected pack)")
#         st.write(f"Using pack: `{PACK_DIR}`")
#
#         run_eval = st.checkbox("Also generate questions + self-eval", value=True)
#         eval_limit = st.slider("Eval question limit", min_value=50, max_value=3000, value=600, step=50)
#
#         if st.button("Build now", type="primary"):
#             br = build_all(ROOT, PACK_DIR, run_eval=run_eval, eval_limit=eval_limit)
#             if br.ok:
#                 st.success(br.message)
#                 st.json(br.stats)
#             else:
#                 st.error(br.message)
#
#         st.divider()
#         st.write("Index files status:")
#         st.write(f"- ontology.json exists: `{(INDEX_DIR / 'ontology.json').exists()}`")
#         st.write(f"- claims.jsonl exists: `{(INDEX_DIR / 'claims.jsonl').exists()}`")
#
#     with tab_ask:
#         st.subheader("3) Ask (requires index built)")
#         if not (INDEX_DIR / "ontology.json").exists():
#             st.warning("Index not built yet. Go to **Build index** and click **Build now**.")
#         else:
#             q = st.text_area(
#                 "Question",
#                 height=120,
#                 value="Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia."
#             )
#
#             if st.button("Answer", type="primary"):
#                 # IMPORTANT: pass the loaded DomainPack
#                 res = answer(INDEX_DIR, q, dp)
#                 c1, c2 = st.columns([1, 1])
#                 with c1:
#                     st.markdown(res.markdown)
#                 with c2:
#                     st.subheader("Debug")
#                     st.json(res.debug)
#
#     with tab_pack:
#         st.subheader("Domain Pack Viewer")
#         st.write(f"Pack dir: `{PACK_DIR}`")
#         st.json(dp.manifest)
#
#         with st.expander("Frames"):
#             st.json(dp.frames)
#
#         with st.expander("Routing rules"):
#             st.json(dp.routing_rules)
#
#         with st.expander("Edge patterns (raw)"):
#             st.json(dp.edge_patterns_raw)
#
#         with st.expander("Infer rules (raw)"):
#             st.json(dp.infer_rules_raw)
#
#         with st.expander("Junk filters"):
#             st.json(dp.junk_filters)
#
#         with st.expander("Scoring"):
#             st.json(dp.scoring)
#
#
# if __name__ == "__main__":
#     main()
#
# # import streamlit as st
# # from pathlib import Path
# # import json
# #
# # from engine.pipeline import ensure_dirs, save_uploaded_docs, build_all
# # from engine.domain_pack import DomainPack
# # from engine.qa import answer
# #
# # ROOT = Path(".").resolve()
# # paths = ensure_dirs(ROOT)
# # DOCS_DIR = paths["docs_dir"]
# # INDEX_DIR = paths["index_dir"]
# # EVAL_DIR = paths["eval_dir"]
# # PACKS_DIR = ROOT / "domain_packs"
# # # ROOT = Path(__file__).parent
# # # PACKS_DIR = ROOT / "domain_packs"
# # PACK_DIR = PACKS_DIR / "gdm_v1"   # default
# #
# # build_all(ROOT, PACK_DIR, ...)
# #
# #
# # st.set_page_config(page_title="GDM ChainGraph — Domain Packs", layout="wide")
# # st.title("GDM ChainGraph — Domain Packs (external frames/patterns/routing)")
# #
# # def list_packs():
# #     if not PACKS_DIR.exists():
# #         return []
# #     out = []
# #     for p in PACKS_DIR.iterdir():
# #         if p.is_dir() and (p / "manifest.json").exists():
# #             out.append(p)
# #     return sorted(out, key=lambda x: x.name)
# #
# # packs = list_packs()
# # if not packs:
# #     st.error("No domain packs found. Create: domain_packs/gdm_v1/manifest.json etc.")
# #     st.stop()
# #
# # pack_names = [p.name for p in packs]
# # sel = st.sidebar.selectbox("Select Domain Pack", pack_names, index=0)
# # PACK_DIR = PACKS_DIR / sel
# #
# # # quick pack info
# # try:
# #     dp = DomainPack.load(PACK_DIR)
# #     st.sidebar.write(f"**Pack:** {dp.manifest.get('name')} v{dp.manifest.get('version')}")
# # except Exception as e:
# #     st.sidebar.error(f"Pack load error: {e}")
# #     st.stop()
# #
# # tab_upload, tab_build, tab_ask, tab_pack = st.tabs(["Upload docs", "Build index", "Ask", "Pack viewer"])
# #
# # with tab_upload:
# #     st.subheader("1) Upload your .txt documents")
# #     st.write(f"Docs will be saved into: `{DOCS_DIR}`")
# #
# #     uploaded = st.file_uploader("Upload .txt docs", type=["txt"], accept_multiple_files=True)
# #     overwrite = st.checkbox("Overwrite if same filename exists", value=True)
# #
# #     if st.button("Save uploaded docs", type="primary"):
# #         res = save_uploaded_docs(DOCS_DIR, uploaded, overwrite=overwrite)
# #         st.success(f"Saved: {len(res['saved'])}, Skipped: {len(res['skipped'])}")
# #         if res["saved"]:
# #             st.write("Saved files:")
# #             for p in res["saved"]:
# #                 st.code(p)
# #         if res["skipped"]:
# #             st.write("Skipped:")
# #             st.json(res["skipped"])
# #
# #     st.divider()
# #     st.write("Currently in docs/:")
# #     files = sorted([p.name for p in DOCS_DIR.glob("*.txt")])
# #     st.dataframe([{"file": f} for f in files] if files else [{"file": "(none)"}])
# #
# # with tab_build:
# #     st.subheader("2) Build / Rebuild index & ontology using selected Domain Pack")
# #     st.write(f"Using pack: `{PACK_DIR}`")
# #
# #     run_eval = st.checkbox("Also generate questions + self-eval", value=True)
# #     eval_limit = st.slider("Eval question limit", min_value=50, max_value=3000, value=600, step=50)
# #
# #     if st.button("Build now", type="primary"):
# #         br = build_all(ROOT, PACK_DIR, run_eval=run_eval, eval_limit=eval_limit)
# #         if br.ok:
# #             st.success(br.message)
# #             st.json(br.stats)
# #         else:
# #             st.error(br.message)
# #
# #     st.divider()
# #     st.write("Index files status:")
# #     st.write(f"- ontology.json: `{(INDEX_DIR/'ontology.json').exists()}`")
# #     st.write(f"- claims.jsonl: `{(INDEX_DIR/'claims.jsonl').exists()}`")
# #
# # with tab_ask:
# #     st.subheader("3) Ask (requires index built)")
# #     if not (INDEX_DIR / "ontology.json").exists():
# #         st.warning("Index not built yet. Go to **Build index** and click **Build now**.")
# #     else:
# #         q = st.text_area(
# #             "Question",
# #             height=120,
# #             value="Based on the evidence presented, should GDM be reframed as a chronic metabolic condition rather than a transient pregnancy complication? Justify your reasoning."
# #         )
# #         if st.button("Answer", type="primary"):
# #             res = answer(INDEX_DIR, q, dp)
# #             c1, c2 = st.columns([1, 1])
# #             with c1:
# #                 st.markdown(res.markdown)
# #             with c2:
# #                 st.subheader("Debug")
# #                 st.json(res.debug)
# #
# # with tab_pack:
# #     st.subheader("Domain Pack Viewer")
# #     st.write(f"Pack dir: `{PACK_DIR}`")
# #     st.json(dp.manifest)
# #
# #     with st.expander("Frames"):
# #         st.json(dp.frames)
# #
# #     with st.expander("Routing rules"):
# #         st.json(dp.routing_rules)
# #
# #     with st.expander("Edge patterns (raw)"):
# #         st.json(dp.edge_patterns_raw)
# #
# #     with st.expander("Infer rules (raw)"):
# #         st.json(dp.infer_rules_raw)
# #
# #     with st.expander("Junk filters"):
# #         st.json(dp.junk_filters)
# #
# #     with st.expander("Scoring"):
# #         st.json(dp.scoring)
