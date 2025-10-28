from sqlalchemy.orm import Session
from src.database import models

def get_all_reports(db: Session):
    return db.query(models.Report).all()

def create_report(db: Session, clss: str, trust: float, treatment: str):
    report = models.Report(clss=clss, trust=trust, treatment=treatment)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report

def delete_report(db: Session, report_id: int):
    report = db.query(models.report).filter(models.report.id == report_id).first()
    if report:
        db.delete(report)
        db.commit()
        return True
    return False
