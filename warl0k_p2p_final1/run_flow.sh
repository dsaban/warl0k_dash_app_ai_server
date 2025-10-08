# Start hub
#python3 hub_server.py

# Enroll B (or run peer_b which enrolls automatically)
python3 check_enroll_peer.py --device-id device-B

# Prepare adapters (pretrain)
# On A side (adapters_A) prepare B's ticket n=1
python3 check_ticket.py --peer-id device-B --n 1 --adapters .adapters_A --mode online --window 1

# On B side (adapters_B) prepare A's ticket n=1
python3 check_ticket.py --peer-id device-A --n 1 --adapters .adapters_B --mode online --window 1

python3 check_peer_flow.py --A device-A --B device-B --n 1 --adapters-A .adapters_A --adapters-B .adapters_B --mode online
