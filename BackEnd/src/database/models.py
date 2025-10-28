from sqlalchemy import Column, Integer, String, Float
from src.database.connection import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    clss = Column(String, nullable=False)
    trust = Column(Float, nullable=False)
    treatment = Column(String, nullable=False)
    