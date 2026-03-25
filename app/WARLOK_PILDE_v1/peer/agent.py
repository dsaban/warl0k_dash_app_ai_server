from core.crypto import encrypt

def send(key, msg):
    return encrypt(key, msg.encode())