from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str
    unit_code: str


class ChatMessage(BaseModel):
    role: str
    content: str


class MedicineChatRequest(BaseModel):
    session_id: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)
    message: str = ""
    use_pubmed: bool = True


class MedicineAddRequest(BaseModel):
    name: str
    generic: str = ""
    form: str = ""
    category: str = "อื่นๆ"
    indications: str = ""
    quantity: int = 0
    unit: str = "หน่วย"


class QuantityUpdateRequest(BaseModel):
    quantity: int


class BBox(BaseModel):
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0


class WoundAnalyzeRequest(BaseModel):
    image_base64: str
    bbox: BBox = Field(default_factory=BBox)
    user_note: str = ""
    reference_cm: Optional[float] = None
    use_pubmed: bool = True
    session_id: Optional[str] = None


class ConsentSaveRequest(BaseModel):
    session_id: str
    mode: str
    consent_given: bool
    transcript: List[ChatMessage] = Field(default_factory=list)
    outcome: str = ""
    referred: bool = False
    red_flag_ids: List[str] = Field(default_factory=list)
