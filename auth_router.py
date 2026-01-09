from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from auth_service import create_user, authenticate, update_user
from jwt_service import create_token

router = APIRouter(prefix="/auth", tags=["Auth"])

class Signup(BaseModel):
    name: str
    email: str
    gender: str
    nationality: str
    password: str

class Login(BaseModel):
    email: str
    password: str

class UpdateProfile(BaseModel):
    name: str | None = None
    gender: str | None = None
    nationality: str | None = None

@router.post("/signup")
def signup(req: Signup):
    if not create_user(req.dict()):
        raise HTTPException(409, "User already exists")
    token = create_token({
        "RowKey": req.email,
        "name": req.name,
        "gender": req.gender,
        "nationality": req.nationality
    })
    return {"access_token": token, "expires_in": 3600}

@router.post("/login")
def login(req: Login):
    user = authenticate(req.email, req.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_token(user), "expires_in": 3600}

@router.put("/profile/{email}")
def update(email: str, req: UpdateProfile):
    update_user(email, {k: v for k, v in req.dict().items() if v})
    return {"updated": True}
