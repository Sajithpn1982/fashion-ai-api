import time
import faiss
import numpy as np
from config import VECTOR_DIM, STM_TTL

_stm_index = faiss.IndexFlatIP(VECTOR_DIM)
_stm_ts = []
_stm_text = []

def stm_cleanup():
    global _stm_index, _stm_ts, _stm_text
    now = time.time()
    keep = [i for i, t in enumerate(_stm_ts) if now - t < STM_TTL]
    if len(keep) == len(_stm_ts):
        return

    new_idx = faiss.IndexFlatIP(VECTOR_DIM)
    new_ts, new_text = [], []
    for i in keep:
        new_idx.add(_stm_index.reconstruct(i).reshape(1, -1))
        new_ts.append(_stm_ts[i])
        new_text.append(_stm_text[i])

    _stm_index, _stm_ts, _stm_text = new_idx, new_ts, new_text

def store_stm(vec, text):
    _stm_index.add(vec)
    _stm_ts.append(time.time())
    _stm_text.append(text)

def latest_vec():
    if _stm_index.ntotal == 0:
        return None
    return _stm_index.reconstruct(_stm_index.ntotal - 1).reshape(1, -1)
