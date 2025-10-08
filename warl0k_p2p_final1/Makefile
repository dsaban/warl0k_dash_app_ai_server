PY=python3

run-hub:
	$(PY) hub_server.py

run-a:
	$(PY) peer_a.py

run-b:
	$(PY) peer_b.py

test:
	$(PY) check_10_sessions.py

# Quick tools
#tools-check-ticket:
#	$(PY) check_ticket.py --peer-id device-A --n 1 --adapters .adapters_B --mode online

tools-check-ticket:
	$(PY) check_ticket.py --peer-id device-B --n 1 --adapters .adapters_A --mode online

tools-check-tag:
	$(PY) check_tag.py --A device-A --B device-B --n 1 --challengeB-hex deadbeef --obf-real cafe1234 --tag-hex 00

tools-clean:
	$(PY) check_clean_adapters.py --path .adapters_A || true
	$(PY) check_clean_adapters.py --path .adapters_B || true

tools-sim:
	$(PY) check_simulate_sessions.py --N 10

tools-enroll-A:
	$(PY) check_enroll_peer.py --device-id device-A

tools-enroll-B:
	$(PY) check_enroll_peer.py --device-id device-B

tools-dump-W-A:
	$(PY) check_dump_W.py --device-id device-A

tools-dump-W-B:
	$(PY) check_dump_W.py --device-id device-B

tools-ping-hub:
	$(PY) check_ping_hub.py

tools-list-adapters-A:
	$(PY) check_list_adapters.py --path .adapters_A
tools-list-adapters-B:
	$(PY) check_list_adapters.py --path .adapters_B

#tools/force_retrain:A:
#	$(PY) check_force_retrain.py --peer-id device-A --n 1 --adapters .adapters_B
#tools/force_retrain:B:
#	$(PY) check_force_retrain.py --peer-id device-B --n 1 --adapters .adapters_A

# new test of learning route
#Start hub:
	$(PY) hub_server.py

# Enroll B (or run peer_b which enrolls automatically)
	$(PY) check_enroll_peer.py --device-id device-B

# Prepare adapters (pretrain)
# On A side (adapters_A) prepare B's ticket n=1
	$(PY) check_ticket.py --peer-id device-B --n 1 --adapters .adapters_A --mode online --window 1

# On B side (adapters_B) prepare A's ticket n=1
	$(PY) check_ticket.py --peer-id device-A --n 1 --adapters .adapters_B --mode online --window 1

tools/peer_flow:
	$(PY) check_peer_flow.py --A device-A --B device-B --n 1 --adapters-A .adapters_A --adapters-B .adapters_B --mode online

tools-list-audits-A:
	$(PY) check_list_audits.py --dir .adapters_A

tools-list-audits-B:
	$(PY) check_list_audits.py --dir .adapters_B
