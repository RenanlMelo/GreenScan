from pydantic import BaseModel

class ReportCreate(BaseModel):
    clss: str
    trust: float
    treatment: str

class ReportResponse(ReportCreate):
    id: int

    class Config:
        orm_mode = True
