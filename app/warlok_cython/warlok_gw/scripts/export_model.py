# scripts/export_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Export trained weights from the Streamlit app to a gateway-ready .npz file.
#
# Two ways to use:
#
#   A) From inside the Streamlit Tab 5 UI (add a button):
#      ────────────────────────────────────────────────────
#      from scripts.export_model import export_from_streamlit
#      if st.button("Export model to gateway"):
#          path = export_from_streamlit(st.session_state.t5_model)
#          st.success(f"Saved → {path}")
#
#   B) Command-line (if you saved a checkpoint externally):
#      ─────────────────────────────────────────────────────
#      python scripts/export_model.py --input checkpoint.pkl --output model.npz
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import argparse, pickle, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from warlok_gw.model_io import save_weights, validate, fingerprint


def export_from_streamlit(params: dict,
                          output: str = "model.npz",
                          threshold: float = 0.82,
                          accuracy:  float | None = None,
                          epochs:    int   | None = None) -> str:
    """
    Export the dict returned by _train() into a .npz ready for the gateway.

    Parameters
    ──────────
    params    : dict returned by _train() in tab5_live_attack_predictor.py
                (same as st.session_state.t5_model)
    output    : output file path (default "model.npz")
    threshold : decision threshold to embed in metadata
    accuracy  : optional session accuracy to embed in metadata
    epochs    : number of training epochs to embed in metadata

    Returns path string so the UI can display it.
    """
    validate(params)

    meta = {"threshold": threshold}
    if accuracy is not None: meta["accuracy"]   = accuracy
    if epochs   is not None: meta["epochs"]     = epochs

    path = save_weights(params, output, extra_meta=meta)
    fp   = fingerprint(path)
    print(f"[export] fingerprint SHA-256: {fp}")
    print(f"[export] deploy with:")
    print(f"         scp {path} gateway-host:/opt/warlok/model.npz")
    return str(path)


def _cli():
    ap = argparse.ArgumentParser(description="Export WARL0K model weights")
    ap.add_argument("--input",  required=True, help="Pickle file containing params dict")
    ap.add_argument("--output", default="model.npz", help="Output .npz path")
    ap.add_argument("--threshold", type=float, default=0.82)
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        params = pickle.load(f)

    export_from_streamlit(params, args.output, args.threshold)


if __name__ == "__main__":
    _cli()
