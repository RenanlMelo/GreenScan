from sqlalchemy.orm import Session
from src.database import models

def get_all_reports(db: Session):
    return db.query(models.Report).all()

def get_report_by_id(db: Session, report_id: int):
    return db.query(models.Report).filter(models.Report.id == report_id).first()

def create_report(
    db: Session,
    clss: str,
    trust: float,
    title: str,
    description: str,
    treatment: str,
    prevention: str,
    image: str
):
    report = models.Report(
        clss=clss,
        trust=trust,
        title=title,
        description=description,
        treatment=treatment,
        prevention=prevention,
        image=image
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def delete_report(db: Session, report_id: int):
    report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if report:
        db.delete(report)
        db.commit()
        return True
    return False
