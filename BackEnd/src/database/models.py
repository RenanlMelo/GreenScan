from sqlalchemy import Column, Integer, String, Float, DateTime
from src.database.connection import Base
from sqlalchemy.sql import func

from datetime import datetime
from pytz import timezone

BR_TZ = timezone("America/Sao_Paulo")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    clss = Column(String, nullable=False)
    trust = Column(Float, nullable=False)
    title = Column(String, nullable=True)
    description = Column(String, nullable=True)
    treatment = Column(String, nullable=False)
    prevention = Column(String, nullable=True)
    image = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(BR_TZ)
    )

