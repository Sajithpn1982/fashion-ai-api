import json
from fastapi import HTTPException
from typing import Optional, Dict, List

def parse_filters(filters: Optional[str]) -> Optional[Dict[str, str]]:
    if not filters:
        return None
    try:
        parsed = json.loads(filters)
        if not isinstance(parsed, dict):
            raise ValueError
        return parsed
    except Exception:
        raise HTTPException(status_code=400, detail="filters must be a valid JSON object")

def extract_all_strings(obj) -> List[str]:
    if isinstance(obj, dict):
        return sum([extract_all_strings(v) for v in obj.values()], [])
    if isinstance(obj, list):
        return sum([extract_all_strings(v) for v in obj], [])
    if isinstance(obj, str):
        return [obj]
    return []

def normalize_terms(text: str) -> List[str]:
    return [t.strip().lower() for t in text.replace("/", ",").split(",") if t.strip()]

def apply_hard_filters(results, filters):
    if not filters:
        return results

    filtered = []
    for item in results:
        searchable_text = [s.lower() for s in extract_all_strings(item)]
        keep = True
        for _, user_value in filters.items():
            terms = normalize_terms(user_value)
            if not any(any(term in t for t in searchable_text) for term in terms):
                keep = False
                break
        if keep:
            filtered.append(item)
    return filtered
