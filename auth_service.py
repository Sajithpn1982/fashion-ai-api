from passlib.context import CryptContext
from datetime import datetime
from table_client import users_table
import hashlib

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
PK = "USER"

def normalize_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(user: dict) -> bool:
    try:
        users_table.get_entity(PK, user["email"])
        return False
    except Exception:
        users_table.create_entity({
            "PartitionKey": PK,
            "RowKey": user["email"],
            "name": user["name"],
            "gender": user["gender"],
            "nationality": user["nationality"],
            "password": pwd.hash(normalize_password(user["password"])),
            "createdAt": datetime.utcnow().isoformat()
        })
        return True

def update_user(email: str, updates: dict):
    entity = users_table.get_entity(PK, email)
    entity.update(updates)
    users_table.update_entity(entity)

def authenticate(email: str, password: str):
    try:
        u = users_table.get_entity(PK, email)
        if pwd.verify(normalize_password(password), u["password"]):
            return u
    except Exception:
        pass
    return None
