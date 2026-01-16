import hashlib
from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from config import JWT_SECRET, JWT_ALG, USER_ID_CLAIM

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
    import uuid
    return x_thread_id or str(uuid.uuid4())
