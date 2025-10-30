from pydantic import BaseModel

class ReportBase(BaseModel):
    clss: str
    trust: float
    treatment: str
    image: str

class ReportCreate(ReportBase):
    pass

class ReportResponse(ReportBase):
    id: int

    class Config:
        orm_mode = True
