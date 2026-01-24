# from warl0k_cloud_demo_app_multi_client_server_dash.warlok_pim_hash_dual import (DualPIMHashConfig, PIMHashDualModel, VerificationError)
#
# # 1. Configure and train model for a single master secret (per device/identity)
# cfg = DualPIMHashConfig(
#     secret_dim=32,
#     obf_dim=32,
#     obf_hidden_dim=64,
#     recon_hidden_dim=64,
#     learning_rate=0.01,
#     epochs=500,
#     window_size=8,
#     noise_std=0.01,   # simulate jitter / PQ stress
# )
#
# model = PIMHashDualModel(cfg)
#
# master_secret = b"this is the device master secret"
# print("Master secret (hex):", master_secret.hex())
# model.fit(master_secret, seed_range=(1, 4), counter_range=(1, 16))
#
# # 2. Device side: generate a temp obfuscated secret for (seed, counter)
# seed = 2
# counter = 7
# window_start = 5
#
# temp_secret = model.generate_obfuscated_secret(seed=seed, counter=counter)
# print("Temp obfuscated secret (hex):", temp_secret.hex())
#
# # 3. Verifier side: verify the obfuscated secret
# try:
#     result = model.verify(
#         obfuscated_secret=temp_secret,
#         seed=seed,
#         counter=counter,
#         window_start=window_start,
#         tolerance=1e-3,
#     )
#     print(f"Verification result: {result}")
#     print("OK:", result.ok, "error=", result.error)
# except VerificationError as e:
#     print("Verification failed:", e)
#
# # 4. Monitor behavior across a chain of counters
# chain = model.monitor_chain(
#     seed=seed,
#     start_counter=4,
#     length=12,
#     window_start=window_start,
#     tolerance=1e-3,
# )
# for r in chain:
#     print(f"ctr={r.counter} "
#           f"ok={r.ok} "
#           f"err={r.error:.6f} "
#           f"window=[{r.window_start},{r.window_end}]")
from warl0k_cloud_demo_app_multi_client_server_dash.warlok_pim_hash_dual import (
    DualPIMHashConfig,
    PIMHashDualModel,
    VerificationError,
)

def main():
    # 1. Configure and train model for a single master secret (per device/identity)
    cfg = DualPIMHashConfig(
        secret_dim=32,
        obf_dim=32,
        obf_hidden_dim=64,
        recon_hidden_dim=64,
        learning_rate=0.01,
        epochs=1000,
        window_size=8,
        noise_std=0.01,   # simulate jitter / PQ stress
    )

    model = PIMHashDualModel(cfg)

    master_secret = b"this is the device master secret"
    print("=== TRAINING MODEL ON MASTER SECRET ===")
    print("Master secret (raw) :", master_secret)
    print("Master secret (hex) :", master_secret.hex())
    model.fit(master_secret, seed_range=(1, 4), counter_range=(1, 16))

    # 2. Single roundtrip: obfuscation -> reconstruction
    seed = 2
    counter = 7
    window_start = 5
    
    print("\n=== ROUNDTRIP (RAW MASTER RECONSTRUCTION) ===")
    obf_bytes, master_hat_vec, master_hat_raw, mse_err = model.roundtrip(seed, counter)
    
    print("Original master secret (raw):")
    print(master_secret)
    print("Original (hex):", master_secret.hex())
    
    print("\nReconstructed master secret (RAW-LIKE):")
    print(master_hat_raw)
    print("Reconstructed (hex):", master_hat_raw.hex())
    
    print("\nVector reconstruction sample (first 8 dims):")
    print(master_hat_vec[:8])
    
    print(f"\nMSE error vs true master vector: {mse_err:.8f}")
    
    # print("\n=== SINGLE ROUNDTRIP (OBFUSCATION -> RECONSTRUCTION) ===")
    # obf_bytes, master_hat, mse_err = model.roundtrip(seed=seed, counter=counter)
    #
    # print(f"Seed={seed}, Counter={counter}")
    # print("Obfuscated secret (bytes len):", len(obf_bytes))
    # print("Obfuscated secret (hex)      :", obf_bytes.hex())
    #
    # # Just show first few dims for readability
    # print("Reconstructed master vec (first 8 dims):")
    #
    # # secret reconstructed (first 8 bytes):
    # print(master_hat[:8])
    # print(f"Reconstruction MSE vs internal master identity: {mse_err:.8f}")

    # 3. Verifier-style check using verify()
    print("\n=== VERIFIER CHECK (with sliding window) ===")
    try:
        result = model.verify(
            obfuscated_secret=obf_bytes,
            seed=seed,
            counter=counter,
            window_start=window_start,
            tolerance=1e-3,
        )
        print("Verification OK")
        print(" -> ok        :", result.ok)
        print(" -> error     :", result.error)
        print(" -> counter   :", result.counter)
        print(" -> window    :", f"[{result.window_start}, {result.window_end}]")
    except VerificationError as e:
        print("Verification FAILED:", e)

    # 4. Monitor behavior across a chain of counters
    print("\n=== CHAIN MONITORING (counters across window) ===")
    chain = model.monitor_chain(
        seed=seed,
        start_counter=4,
        length=12,
        window_start=window_start,
        tolerance=1e-3,
    )
    for r in chain:
        print(
            f"ctr={r.counter:2d} "
            f"ok={str(r.ok):5s} "
            f"err={r.error:.6f} "
            f"window=[{r.window_start},{r.window_end}]"
        )

if __name__ == "__main__":
    main()
