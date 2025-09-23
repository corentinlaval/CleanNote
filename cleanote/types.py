# cleanote/types.py
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


class Doc(BaseModel):
    id: str
    text: str
    meta: Dict[str, Any] = {}


class Context(BaseModel):
    run_id: str
    params: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}  # rempli par ModelLoader


class Issue(BaseModel):
    code: str
    message: str
    severity: str  # "info" | "warn" | "error"
    span: Optional[Tuple[int, int]] = None


class Report(BaseModel):
    doc_id: str
    issues: List[Issue] = []
