import hashlib

def hash_block(prev, data):
    return hashlib.sha256(prev + data).digest()