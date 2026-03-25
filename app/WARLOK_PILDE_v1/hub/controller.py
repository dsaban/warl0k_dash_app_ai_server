from hsm.demo_hsm import DemoHSM

hsm = DemoHSM(b'secret')

def get_epoch(epoch_id):
    return hsm.derive_key(b'epoch', epoch_id.to_bytes(4,'big'))