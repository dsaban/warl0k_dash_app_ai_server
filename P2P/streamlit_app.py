# streamlit_app.py
# ------------------------------------
# WARL0K P2P Mock Demo (Streamlit)
# Three columns: Peer A | Hub Log | Peer B
# - Simulates rendezvous, dual-seed init, nano-AI key derivation, policy checks,
#   LinkObject headers, and envelope AEAD (MOCK only).
# - Shows message/file/payment flows and a central log (no plaintext in hub).
#
# DISCLAIMER: This is a MOCK cryptographic demo.
# Do NOT use this code for production cryptography.
#
# Run: streamlit run streamlit_app.py
# ------------------------------------
from __future__ import annotations

import streamlit as st
import hashlib, hmac, secrets, uuid, json, base64, time
from datetime import datetime
from io import BytesIO

# ============= MOCK CRYPTO PRIMITIVES (NOT SECURE) =============
def hkdf(key: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """Simple HKDF-like expander using HMAC-SHA256 (MOCK)."""
    okm = b""
    prev = b""
    counter = 1
    while len(okm) < length:
        prev = hmac.new(salt, prev + info + bytes([counter]), hashlib.sha256).digest()
        okm += prev
        counter += 1
    return okm[:length]

def mock_nano_model(session_id: str, seed: bytes, counter: int, drift: bytes=b"") -> tuple[bytes, int]:
    """
    Deterministic key derivation (MOCK) to emulate nano-AI:
    K_i = HMAC(seed, session_id || counter || drift), FP_i = lower 32 bits of digest
    """
    msg = session_id.encode() + counter.to_bytes(8, "big") + drift
    digest = hmac.new(seed, msg, hashlib.sha256).digest()
    K_i = hkdf(digest, salt=seed, info=b"WARLOK-KDF", length=32)
    fp_hint = int.from_bytes(digest[:4], "big")
    return K_i, fp_hint

def mock_aead_seal(key: bytes, nonce: bytes, aad: bytes, plaintext: bytes) -> tuple[bytes, bytes]:
    """
    MOCK AEAD: XOR "stream cipher" via HMAC-derived keystream + HMAC tag over (aad||cipher||nonce).
    Not secure; for visualization only.
    """
    # derive keystream
    stream = hmac.new(key, nonce + b"STREAM", hashlib.sha256).digest()
    # extend keystream to length of plaintext
    ks = (stream * ((len(plaintext) // len(stream)) + 1))[:len(plaintext)]
    ciphertext = bytes([p ^ k for p, k in zip(plaintext, ks)])
    tag = hmac.new(key, aad + ciphertext + nonce, hashlib.sha256).digest()
    return ciphertext, tag

def mock_aead_open(key: bytes, nonce: bytes, aad: bytes, ciphertext: bytes, tag: bytes) -> bytes | None:
    calc = hmac.new(key, aad + ciphertext + nonce, hashlib.sha256).digest()
    if not hmac.compare_digest(calc, tag):
        return None
    stream = hmac.new(key, nonce + b"STREAM", hashlib.sha256).digest()
    ks = (stream * ((len(ciphertext) // len(stream)) + 1))[:len(ciphertext)]
    plaintext = bytes([c ^ k for c, k in zip(ciphertext, ks)])
    return plaintext

def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()

# ============= LINK OBJECT (headers only) =============
def make_link_object(link_type: str, to_pdid: str, from_pdid: str, meta: dict, content_ref: str | None, policy_ref: str, sk_auth: bytes) -> dict:
    core = {
        "link_id": str(uuid.uuid4()),
        "type": link_type,
        "to": to_pdid,
        "from": from_pdid,
        "meta": meta,
        "content_ref": content_ref,
        "policy_ref": policy_ref,
    }
    core_bytes = json.dumps(core, sort_keys=True).encode()
    sender_sig = hmac.new(sk_auth, core_bytes, hashlib.sha256).hexdigest()
    core["proof"] = {"sender_sig": sender_sig}
    return core

def verify_link_object(lo: dict, pk_auth: bytes) -> bool:
    # In this mock, pk_auth==sk_auth just for demo (no real public-key scheme here).
    temp = lo.copy()
    proof = temp.pop("proof", None)
    if not proof or "sender_sig" not in proof:
        return False
    data = json.dumps(temp, sort_keys=True).encode()
    expect = hmac.new(pk_auth, data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expect, proof["sender_sig"])

# ============= HUB / POLICY (MOCK) =============
def policy_check(header: dict, fp_hint: int) -> dict:
    """Simple allow/block based on size/amount thresholds & fp_hint sanity."""
    t = header.get("type")
    meta = header.get("meta", {})
    ok = True
    reason = "ok"
    if t == "file":
        size = int(meta.get("size", 0))
        if size > st.session_state.get("policy_file_max", 50_000_000):  # 50MB default
            ok, reason = False, "file too large"
    if t == "msg":
        l = int(meta.get("chars", 0))
        if l > st.session_state.get("policy_msg_max_chars", 5000):
            ok, reason = False, "message too long"
    if t == "pay":
        amt = float(meta.get("amount", 0.0))
        if amt > st.session_state.get("policy_pay_max", 10_000.0):
            ok, reason = False, "amount exceeds policy"
    # basic fp_hint sanity (mock anomaly): zero hint flagged as anomaly
    anomaly = (fp_hint == 0)
    ticket = {
        "ok": ok,
        "reason": reason,
        "anomaly": anomaly,
        "ticket": {
            "sid": st.session_state["sid"],
            "link_id": header["link_id"],
            "ttl": int(time.time()) + 120
        } if ok else None
    }
    return ticket

def hub_log(event_type: str, payload: dict):
    entry = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "type": event_type,
        "payload": payload
    }
    st.session_state["hub_logs"].append(entry)

# ============= SESSION STATE INIT =============
def init_state():
    ss = st.session_state
    ss.setdefault("hub_logs", [])
    ss.setdefault("sid", None)
    ss.setdefault("ms_seed", None)
    ss.setdefault("os_seed", None)
    ss.setdefault("counter", 0)
    ss.setdefault("policy_file_max", 50_000_000)
    ss.setdefault("policy_msg_max_chars", 5000)
    ss.setdefault("policy_pay_max", 10_000.0)
    # pseudo auth keys for peers (mock HMAC keys used as both "pk" and "sk")
    ss.setdefault("peerA_auth", secrets.token_bytes(32))
    ss.setdefault("peerB_auth", secrets.token_bytes(32))
    ss.setdefault("peerA_pdid", "did:pA:" + secrets.token_hex(4))
    ss.setdefault("peerB_pdid", "did:pB:" + secrets.token_hex(4))

def new_session():
    st.session_state["sid"] = str(uuid.uuid4())
    st.session_state["ms_seed"] = secrets.token_bytes(32)
    st.session_state["os_seed"] = secrets.token_bytes(32)
    st.session_state["counter"] = 0
    hub_log("rendezvous", {"sid": st.session_state["sid"]})
    hub_log("seeds_issued", {"sid": st.session_state["sid"], "ms_seed": b64(st.session_state["ms_seed"]), "os_seed": b64(st.session_state["os_seed"])})

# ============= UI HELPERS =============
def step_box(title: str, lines: list[str]):
    st.markdown(f"**{title}**")
    for ln in lines:
        st.write("• " + ln)

def render_logs():
    st.markdown("### Hub Log (headers & decisions only; no payloads)")
    # render last 200 events
    logs = st.session_state["hub_logs"][-200:]
    for e in reversed(logs):
        st.code(json.dumps(e, indent=2))

# ============= MAIN APP =============
def main():
    st.set_page_config(page_title="WARL0K P2P Mock", layout="wide")
    init_state()

    st.title("WARL0K P2P Mock Demo — Peers, Hub, and Vanishing Secrets")
    st.caption("Education demo. Not real cryptography. Shows envelopes, LinkObject headers, policy checks, and a central Hub log.")

    top_cols = st.columns([1,1,1])
    with top_cols[0]:
        if st.button("🔄 New Session"):
            new_session()
    with top_cols[1]:
        st.slider("Policy: Max file size (bytes)", 100_000, 500_000_000, key="policy_file_max")
    with top_cols[2]:
        st.slider("Policy: Max payment amount", 10.0, 100_000.0, key="policy_pay_max")

    st.divider()

    colA, colHub, colB = st.columns([1.1, 1.2, 1.1])

    # ------------------ PEER A ------------------
    with colA:
        st.subheader("Peer A")
        sid = st.session_state["sid"]
        if not sid:
            st.info("Start a session to begin (🔄 New Session).")
        else:
            st.text(f"Session: {sid[:8]}…")
            st.text(f"A pDID: {st.session_state['peerA_pdid']}")
            st.text(f"B pDID: {st.session_state['peerB_pdid']}")
            st.write("—")

            # derive current key & fp from nano-model (MOCK) for A
            ctr = st.session_state["counter"]
            KA, fpA = mock_nano_model(sid, st.session_state["ms_seed"], ctr)
            st.caption(f"Derived K_{ctr} (A) [mock]: {b64(KA)[:16]}… | fp_hint={fpA}")

            with st.expander("Send MESSAGE"):
                msg = st.text_area("Message text", "Hello from A to B")
                if st.button("Send Message ▶️", key="send_msg"):
                    aad = f"{sid}|{ctr}".encode()
                    nonce = secrets.token_bytes(12)
                    ct, tag = mock_aead_seal(KA, nonce, aad, msg.encode())
                    header = make_link_object(
                        "msg",
                        to_pdid=st.session_state["peerB_pdid"],
                        from_pdid=st.session_state["peerA_pdid"],
                        meta={"chars": len(msg), "created_at": datetime.utcnow().isoformat()+"Z"},
                        content_ref=None,
                        policy_ref="default-v1",
                        sk_auth=st.session_state["peerA_auth"]
                    )
                    hub_log("link_register", {"header": header, "fp_hint": fpA})
                    ticket = policy_check(header, fpA)
                    hub_log("policy_decision", {"header": {"link_id": header["link_id"], "type": "msg"}, "ticket": ticket})
                    if ticket["ok"]:
                        # simulate delivery to B
                        st.session_state["last_delivery"] = {
                            "type": "msg",
                            "nonce": nonce,
                            "aad": aad,
                            "ct": ct,
                            "tag": tag,
                            "header": header,
                            "counter": ctr,
                        }
                        st.success("Message sent (mock)")
                        st.session_state["counter"] += 1
                    else:
                        st.error(f"Blocked by policy: {ticket['reason']}")

            with st.expander("Send FILE"):
                up = st.file_uploader("Choose a file", type=None, key="fileA")
                if up and st.button("Send File ▶️", key="send_file"):
                    data = up.read()
                    aad = f"{sid}|{ctr}".encode()
                    nonce = secrets.token_bytes(12)
                    ct, tag = mock_aead_seal(KA, nonce, aad, data)
                    # content_ref = hash of ciphertext for integrity
                    content_ref = hashlib.sha256(ct).hexdigest()
                    header = make_link_object(
                        "file",
                        to_pdid=st.session_state["peerB_pdid"],
                        from_pdid=st.session_state["peerA_pdid"],
                        meta={"size": len(data), "filename": up.name, "mime": up.type, "created_at": datetime.utcnow().isoformat()+"Z"},
                        content_ref=content_ref,
                        policy_ref="default-v1",
                        sk_auth=st.session_state["peerA_auth"]
                    )
                    hub_log("link_register", {"header": header, "fp_hint": fpA})
                    ticket = policy_check(header, fpA)
                    hub_log("policy_decision", {"header": {"link_id": header["link_id"], "type": "file"}, "ticket": ticket})
                    if ticket["ok"]:
                        st.session_state["last_delivery"] = {
                            "type": "file",
                            "nonce": nonce,
                            "aad": aad,
                            "ct": ct,
                            "tag": tag,
                            "header": header,
                            "counter": ctr,
                        }
                        st.success("File sent (mock)")
                        st.session_state["counter"] += 1
                    else:
                        st.error(f"Blocked by policy: {ticket['reason']}")

            with st.expander("Send PAYMENT"):
                amt = st.number_input("Amount", min_value=0.0, value=42.0, step=1.0)
                asset = st.selectbox("Asset", ["USD", "EUR", "USDC", "WBTC"])
                memo = st.text_input("Memo", "Thanks!")
                if st.button("Send Payment ▶️", key="send_pay"):
                    invoice = {
                        "invoice_id": str(uuid.uuid4()),
                        "to": st.session_state["peerB_pdid"],
                        "from": st.session_state["peerA_pdid"],
                        "amount": amt,
                        "asset": asset,
                        "memo": memo,
                        "created_at": datetime.utcnow().isoformat()+"Z",
                        "expiry": int(time.time()) + 3600,
                    }
                    # Inline "invoice" as message payload
                    payload = json.dumps(invoice).encode()
                    aad = f"{sid}|{ctr}".encode()
                    nonce = secrets.token_bytes(12)
                    ct, tag = mock_aead_seal(KA, nonce, aad, payload)
                    header = make_link_object(
                        "pay",
                        to_pdid=st.session_state["peerB_pdid"],
                        from_pdid=st.session_state["peerA_pdid"],
                        meta={"amount": amt, "asset": asset, "memo": memo, "created_at": invoice["created_at"]},
                        content_ref=None,
                        policy_ref="default-v1",
                        sk_auth=st.session_state["peerA_auth"]
                    )
                    hub_log("link_register", {"header": header, "fp_hint": fpA})
                    ticket = policy_check(header, fpA)
                    hub_log("policy_decision", {"header": {"link_id": header["link_id"], "type": "pay"}, "ticket": ticket})
                    if ticket["ok"]:
                        st.session_state["last_delivery"] = {
                            "type": "pay",
                            "nonce": nonce,
                            "aad": aad,
                            "ct": ct,
                            "tag": tag,
                            "header": header,
                            "counter": ctr,
                        }
                        st.success("Payment request sent (mock)")
                        st.session_state["counter"] += 1
                    else:
                        st.error(f"Blocked by policy: {ticket['reason']}")

    # ------------------ HUB ------------------
    with colHub:
        st.subheader("WARL0K Hub (Monitoring & Policy)")
        if st.session_state["sid"]:
            st.text(f"Session: {st.session_state['sid'][:8]}…")
        else:
            st.text("No active session")
        render_logs()

    # ------------------ PEER B ------------------
    with colB:
        st.subheader("Peer B")
        sid = st.session_state["sid"]
        if not sid:
            st.info("Start a session to begin (🔄 New Session).")
        else:
            st.text(f"Session: {sid[:8]}…")
            st.text(f"B pDID: {st.session_state['peerB_pdid']}")
            st.text(f"A pDID: {st.session_state['peerA_pdid']}")
            st.write("—")

            # derive key for B (must match same ctr as delivery)
            last = st.session_state.get("last_delivery")
            if last:
                ctr = last["counter"]
                KB, fpB = mock_nano_model(sid, st.session_state["ms_seed"], ctr)
                st.caption(f"Derived K_{ctr} (B) [mock]: {b64(KB)[:16]}… | fp_hint={fpB}")
                header = last["header"]
                with st.expander("Inspect Link Header"):
                    st.json(header)

                # verify header sig (mock)
                ok_header = verify_link_object(header, st.session_state["peerA_auth"])
                st.write(f"Header signature valid: {'✅' if ok_header else '❌'}")

                # open envelope
                pt = mock_aead_open(KB, last["nonce"], last["aad"], last["ct"], last["tag"])
                if pt is None:
                    st.error("AEAD tag invalid — drop packet")
                    hub_log("delivery_fail", {"reason": "tag_invalid", "header": {"link_id": header["link_id"], "type": header["type"]}})
                else:
                    if last["type"] == "msg":
                        st.success("Message received (mock)")
                        st.code(pt.decode(errors="ignore"))
                    elif last["type"] == "file":
                        st.success("File received (mock)")
                        received_ref = hashlib.sha256(last["ct"]).hexdigest()
                        st.write(f"ContentRef matched: {'✅' if received_ref == header['content_ref'] else '❌'}")
                        st.download_button("Download decrypted file", data=pt, file_name=header['meta'].get("filename","file.bin"))
                    elif last["type"] == "pay":
                        st.success("Payment request received (mock)")
                        inv = json.loads(pt.decode())
                        st.json(inv)
                        # simulate receipt
                        if st.button("Acknowledge Payment (mock)"):
                            receipt = {
                                "invoice_id": inv["invoice_id"],
                                "tx_ref": "psp:DEMO-" + secrets.token_hex(6),
                                "settled_amount": inv["amount"],
                                "asset": inv["asset"],
                                "settled_at": datetime.utcnow().isoformat()+"Z",
                            }
                            hub_log("payment_receipt", {"invoice_id": inv["invoice_id"], "tx_ref": receipt["tx_ref"]})
                            st.success(f"Receipt: {receipt['tx_ref']}")
            else:
                st.info("Awaiting delivery…")

    st.divider()
    st.caption("© WARL0K mock. Vanishing secrets are simulated via deterministic derivation per (sid, seed, counter). AEAD is MOCK.")

if __name__ == "__main__":
    main()
