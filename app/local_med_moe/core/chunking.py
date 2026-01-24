# core/chunking.py
def split_into_chunks(text: str, chunk_size: int, overlap: int):
    text = (text or "").strip()
    if not text:
        return []

    if chunk_size < 200:
        chunk_size = 200
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunk = text[i:j].strip()
        if len(chunk) > 60:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks
