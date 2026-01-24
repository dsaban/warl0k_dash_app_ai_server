class VerificationError(Exception):
    """Raised when secret verification fails (tampering / invalid window / mismatch)."""
    pass
