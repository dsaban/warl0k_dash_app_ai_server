# model/model_wrapper.py
import binascii
from .rnn_model import GatedRNNBroker, TinyModelMCU

# choose vocab for RNN (hex digits)
VOCAB = list("0123456789abcdef")

class SessionModelAgent:
    """
    Local wrapper: given session key bytes -> produce short obf string (hex truncated),
    train small RNN to map obf->identity_string, and verify by predicting.
    """
    def __init__(self, vocab=VOCAB, hidden_dim=32, lr=0.02):
        self.vocab = vocab
        self.rnn = GatedRNNBroker(vocab=vocab, hidden_dim=hidden_dim, lr=lr)
        # optional tiny MCU for device-generated identity samples
        self.mcu = TinyModelMCU(vocab=vocab, hidden_dim=8, seed=123)

    def k_to_obf_string(self, k_bytes: bytes, length: int = 8) -> str:
        # derive a hex string from K_session and truncate to length in hex chars
        h = binascii.hexlify(k_bytes).decode()
        return h[:length]

    def train_on_pair(self, obf_str: str, target_identity: str, epochs: int = 500):
        """
        Train RNN locally to map obf_str -> target_identity by concatenating
        input->target in a simple reconstruction task. This is a toy approach:
        we train the RNN on the target string, assuming obf provides conditioning.
        """
        # Very small trick: train model on target_identity only; in practice you'd train a
        # conditional model. Here it's a demo: we pretend obf_str is the single datum that
        # influences internal weights via repeated tiny epochs.
        # Prepend obf token to target for slight conditioning:
        train_string = obf_str + target_identity
        self.rnn.train(train_string, epochs=epochs)

    def predict_identity(self, obf_str: str) -> str:
        # For this toy model, generate by fingerprint_verify on obf_str + padding
        # then slice the predicted tail as identity
        pred = self.rnn.fingerprint_verify(obf_str + ("0" * len(obf_str)))
        # since training used obf+identity, strip obf prefix
        return pred[len(obf_str):len(obf_str)+8]
