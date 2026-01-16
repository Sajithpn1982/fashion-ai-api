import os
import uuid
import time
import json
import hashlib
import pickle
from typing import Optional, Dict, List

import numpy as np
import torch
import faiss
import open_clip

from PIL import Image
from fastapi import (
    FastAPI,
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends,
    Header,
    HTTPException,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from azure_blob import download_blob, upload_blob, image_url_from_local_path

# =====================================================
# CONFIG
# =====================================================

DEVICE = "cpu"
VECTOR_DIM = 512
TOP_K_RESULTS = 5
STM_TTL = 1800  # 30 minutes

TMP_DIR = "/tmp"
LTM_INDEX_BLOB = "memory/ltm.index"
LTM_META_BLOB = "memory/ltm.meta.pkl"

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = "HS256"
USER_ID_CLAIM = "sub"

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(title="Fashion Recommendation API")
router = APIRouter(prefix="/fashionrec", tags=["Fashion Recommendation"])
app.include_router(router)

# =====================================================
# AUTH + THREAD
# =====================================================

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALG])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    uid = payload.get(USER_ID_CLAIM)
    if not uid:
        raise HTTPException(status_code=401, detail="Missing user id")

    return hashlib.sha256(uid.encode()).hexdigest()

def get_thread_id(x_thread_id: str | None = Header(default=None)) -> str:
    return x_thread_id or str(uuid.uuid4())

# =====================================================
# FILTER PARSING
# =====================================================

def parse_filters(filters: Optional[str]) -> Optional[Dict[str, str]]:
    if not filters:
        return None
    try:
        parsed = json.loads(filters)
        if not isinstance(parsed, dict):
            raise ValueError
        return parsed
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="filters must be a valid JSON object"
        )

# =====================================================
# HARD FILTER (SCHEMA-AGNOSTIC)
# =====================================================

def extract_all_strings(obj) -> List[str]:
    values = []
    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(extract_all_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            values.extend(extract_all_strings(v))
    elif isinstance(obj, str):
        values.append(obj)
    return values

def normalize_terms(text: str) -> List[str]:
    return [
        t.strip().lower()
        for t in text.replace("/", ",").split(",")
        if t.strip()
    ]

def apply_hard_filters(results, filters):
    """
    AND across filters
    OR within each filter
    Schema-agnostic
    """
    if not filters:
        return results

    filtered = []

    for item in results:
        searchable_text = [
            s.lower() for s in extract_all_strings(item)
        ]

        keep = True
        for _, user_value in filters.items():
            expected_terms = normalize_terms(user_value)

            if not any(
                any(term in text for text in searchable_text)
                for term in expected_terms
            ):
                keep = False
                break

        if keep:
            filtered.append(item)

    return filtered

# =====================================================
# GLOBAL STATE
# =====================================================

_model = None
_preprocess = None
_tokenizer = None

_product_index = None
_product_meta = None

_stm_index = None
_stm_ts = []
_stm_text = []

_ltm_index = None
_ltm_meta = []
_ltm_text = []

# =====================================================
# LOADERS
# =====================================================

def load_model():
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    _model.eval().to(DEVICE)
    _tokenizer = open_clip.get_tokenizer("ViT-B-32")

def load_products():
    global _product_index, _product_meta
    if _product_index is not None:
        return

    idx = download_blob("image_embeddings.faiss")
    meta = download_blob("metadata.pkl")

    _product_index = faiss.read_index(idx)
    with open(meta, "rb") as f:
        _product_meta = pickle.load(f)

def load_memory():
    global _stm_index, _ltm_index, _ltm_meta, _ltm_text

    if _stm_index is None:
        _stm_index = faiss.IndexFlatIP(VECTOR_DIM)

    if _ltm_index is None:
        try:
            idx = download_blob(LTM_INDEX_BLOB)
            meta = download_blob(LTM_META_BLOB)
            _ltm_index = faiss.read_index(idx)
            payload = pickle.load(open(meta, "rb"))
            _ltm_meta = payload["meta"]
            _ltm_text = payload["text"]
        except Exception:
            _ltm_index = faiss.IndexFlatIP(VECTOR_DIM)
            _ltm_meta = []
            _ltm_text = []

# =====================================================
# EMBEDDINGS
# =====================================================

def normalize(vec: torch.Tensor) -> np.ndarray:
    vec = vec / torch.clamp(vec.norm(dim=-1, keepdim=True), min=1e-6)
    return vec.cpu().numpy().astype("float32")

def encode_text(text: str) -> np.ndarray:
    tokens = _tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        return normalize(_model.encode_text(tokens))

def encode_image(img: Image.Image) -> np.ndarray:
    img = _preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return normalize(_model.encode_image(img))

# =====================================================
# STM / LTM
# =====================================================

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

def persist_ltm():
    idx_path = os.path.join(TMP_DIR, "ltm.index")
    meta_path = os.path.join(TMP_DIR, "ltm.meta.pkl")

    faiss.write_index(_ltm_index, idx_path)
    pickle.dump({"meta": _ltm_meta, "text": _ltm_text}, open(meta_path, "wb"))

    upload_blob(idx_path, LTM_INDEX_BLOB)
    upload_blob(meta_path, LTM_META_BLOB)

def store_ltm(vec, text):
    _ltm_index.add(vec)
    _ltm_meta.append(time.time())
    _ltm_text.append(text)
    persist_ltm()

def importance_score(text: str) -> int:
    t = text.lower()
    if "remember" in t:
        return 10
    if "favorite" in t:
        return 8
    return 3

def latest_vec(index):
    if index is None or index.ntotal == 0:
        return None
    return index.reconstruct(index.ntotal - 1).reshape(1, -1)

# =====================================================
# MEMORY INTROSPECTION (RICH SUMMARY)
# =====================================================

def memory_usage_trace():
    return {
        "stm": {
            "used": _stm_index is not None and _stm_index.ntotal > 0,
            "count": _stm_index.ntotal if _stm_index else 0,
            "recent_texts": _stm_text[-5:],
            "latest_text": _stm_text[-1] if _stm_text else None,
        },
        "ltm": {
            "used": _ltm_index is not None and _ltm_index.ntotal > 0,
            "count": _ltm_index.ntotal if _ltm_index else 0,
            "stored_texts": _ltm_text[-5:],
            "latest_text": _ltm_text[-1] if _ltm_text else None,
        },
    }

def executed_query_trace(query_type, raw_input, filters):
    mem = memory_usage_trace()
    return {
        "type": query_type,
        "raw_input": raw_input,
        "filters": filters,
        "memory_influence": {
            "stm_used": mem["stm"]["used"],
            "ltm_used": mem["ltm"]["used"],
            "stm_latest_text": mem["stm"]["latest_text"],
            "ltm_latest_text": mem["ltm"]["latest_text"],
        },
        "query_components": {
            "user_input": raw_input is not None,
            "stm_context": mem["stm"]["used"],
            "ltm_context": mem["ltm"]["used"],
            "filters_applied": bool(filters),
        },
    }

# =====================================================
# QUERY VECTOR
# =====================================================

def build_query_vector(input_vec, stm_vec, ltm_vec):
    vec = 0.7 * input_vec
    if stm_vec is not None:
        vec += 0.2 * stm_vec
    if ltm_vec is not None:
        vec += 0.1 * ltm_vec

    vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6, None)
    return vec.astype("float32")

# =====================================================
# PRODUCT RANKING
# =====================================================

def rank_products(vec, k):
    scores, ids = _product_index.search(vec, min(k * 5, _product_index.ntotal))
    results = []

    for s, i in zip(scores[0], ids[0]):
        item = _product_meta[i].copy()
        item["similarity"] = float(s)
        item["image_url"] = image_url_from_local_path(item["image_path"])
        results.append(item)

    return results

# =====================================================
# ROUTES
# =====================================================

@router.post("/recommend/text")
def recommend_text(
    query: str = Form(...),
    filters: Optional[str] = Form(None),
    k: int = TOP_K_RESULTS,
    user_id: str = Depends(get_current_user),
    thread_id: str = Depends(get_thread_id),
):
    filters_dict = parse_filters(filters)

    load_model()
    load_products()
    load_memory()
    stm_cleanup()

    iv = encode_text(query)
    qv = build_query_vector(iv, latest_vec(_stm_index), latest_vec(_ltm_index))

    store_stm(iv, query)
    if importance_score(query) >= 7:
        store_ltm(iv, query)

    raw = rank_products(qv, k)
    results = apply_hard_filters(raw, filters_dict)[:k]

    return {
        "thread_id": thread_id,
        "executed_query": executed_query_trace("text", query, filters_dict),
        "memory": memory_usage_trace(),
        "results": results,
    }

@router.post("/recommend/image")
async def recommend_image(
    file: UploadFile = File(...),
    filters: Optional[str] = Form(None),
    k: int = TOP_K_RESULTS,
    user_id: str = Depends(get_current_user),
    thread_id: str = Depends(get_thread_id),
):
    filters_dict = parse_filters(filters)

    load_model()
    load_products()
    load_memory()
    stm_cleanup()

    img = Image.open(file.file).convert("RGB")
    iv = encode_image(img)
    qv = build_query_vector(iv, latest_vec(_stm_index), latest_vec(_ltm_index))

    store_stm(iv, None)

    raw = rank_products(qv, k)
    results = apply_hard_filters(raw, filters_dict)[:k]

    return {
        "thread_id": thread_id,
        "executed_query": executed_query_trace("image", None, filters_dict),
        "memory": memory_usage_trace(),
        "results": results,
    }

@router.post("/recommend/hybrid")
async def recommend_hybrid(
    query: str = Form(...),
    file: UploadFile = File(...),
    filters: Optional[str] = Form(None),
    k: int = TOP_K_RESULTS,
    user_id: str = Depends(get_current_user),
    thread_id: str = Depends(get_thread_id),
):
    filters_dict = parse_filters(filters)

    load_model()
    load_products()
    load_memory()
    stm_cleanup()

    img = Image.open(file.file).convert("RGB")
    iv = encode_image(img)
    tv = encode_text(query)

    input_vec = (iv + tv) / 2
    input_vec /= np.clip(np.linalg.norm(input_vec, axis=1, keepdims=True), 1e-6, None)

    qv = build_query_vector(
        input_vec, latest_vec(_stm_index), latest_vec(_ltm_index)
    )

    store_stm(input_vec, query)
    if importance_score(query) >= 7:
        store_ltm(input_vec, query)

    raw = rank_products(qv, k)
    results = apply_hard_filters(raw, filters_dict)[:k]

    return {
        "thread_id": thread_id,
        "executed_query": executed_query_trace("hybrid", query, filters_dict),
        "memory": memory_usage_trace(),
        "results": results,
    }
