"""
Simple in-module checks. Run for quick verification.
"""

# from .demo import run_demo
# from warl0k_cloud_demo_app_multi_client_server_dash.P2P.warl0k_p2p import *
from . import demo
run_demo = demo.run_demo
def smoke():
    print("Running smoke demo...")
    run_demo()

if __name__ == "__main__":
    smoke()
