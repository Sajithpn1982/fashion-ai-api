from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from filters_service import save_filters, get_filters

router = APIRouter(prefix="/filters", tags=["User Filters"])

class PriceRange(BaseModel):
    min: float
    max: float

class UserFilters(BaseModel):
    style: Optional[str] = None
    occasion: Optional[str] = None
    size: Optional[str] = None
    colours: Optional[List[str]] = []
    brands: Optional[List[str]] = []
    price: Optional[PriceRange] = None

@router.post("/{email}")
def save(email: str, filters: UserFilters):
    save_filters(email, filters.dict())
    return {"saved": True}

@router.get("/{email}")
def fetch(email: str):
    return get_filters(email)
