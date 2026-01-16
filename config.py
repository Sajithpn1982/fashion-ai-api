import os

DEVICE = "cpu"
VECTOR_DIM = 512
TOP_K_RESULTS = 5
STM_TTL = 1800  # 30 minutes

TMP_DIR = "/tmp"
LTM_INDEX_BLOB = "memory/ltm.index"
LTM_META_BLOB = "memory/ltm.meta.pkl"

# 🔐 JWT CONFIG (THIS WAS MISSING)
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = "HS256"
USER_ID_CLAIM = "sub"
