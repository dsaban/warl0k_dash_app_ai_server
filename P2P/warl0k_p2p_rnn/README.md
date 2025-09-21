# WARL0K Model Demo

A demonstration project where two peers:
- enroll with a hub,
- derive a master_seed locally,
- perform hub-free ephemeral ECDH sessions,
- derive an obfuscation string from the session key,
- locally train a tiny RNN to map that obfuscation string to a device identity string (master proxy),
- use the RNN-predicted identity to validate the session.

Run:
```bash
python demo_model.py
