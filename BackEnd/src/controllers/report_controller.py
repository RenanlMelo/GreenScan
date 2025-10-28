from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.services import report_service
from src.utils.format_response import format_response
from src.schemas.report_schema import ReportCreate, ReportResponse

router = APIRouter()

@router.get("/")
def get_reports(db: Session = Depends(get_db)):
    reports = report_service.get_all_reports(db)
    return format_response(True, "Reports returned successfully", reports)

@router.post("/create", response_model=ReportCreate)
def add_report(report: ReportCreate, db: Session = Depends(get_db)):
    new_report = report_service.create_report(
        db,
        clss=report.clss,
        trust=report.trust,
        treatment=report.treatment
    )
    return new_report

@router.delete("/{report_id}")
def remove_user(report_id: int, db: Session = Depends(get_db)):
    deleted = report_service.delete_report(db, report_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Report not found")
    return format_response(True, "Report deleted successfully")