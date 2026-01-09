from datetime import datetime, timedelta
from jose import jwt
import os

SECRET = os.getenv("JWT_SECRET", "dev-secret")
ALG = "HS256"

def create_token(user: dict):
    payload = {
        "sub": user["RowKey"],
        "name": user["name"],
        "gender": user["gender"],
        "nationality": user["nationality"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET, algorithm=ALG)
