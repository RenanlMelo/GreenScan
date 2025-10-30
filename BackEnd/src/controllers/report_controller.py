from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import shutil
import uuid
import os

from src.utils.format_response import format_response
from src.database.connection import get_db
from src.services import report_service

router = APIRouter()

# Pasta onde as imagens serão salvas
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Monta a pasta como rota estática para acesso público
# Isso deve estar no main.py, mas deixo o comentário para lembrar:
# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@router.get("/")
def get_reports(db: Session = Depends(get_db)):
    reports = report_service.get_all_reports(db)
    return format_response(True, "Reports returned successfully", reports)


@router.get("/{report_id}")
def get_report(report_id: int, db: Session = Depends(get_db)):
    report = report_service.get_report_by_id(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return format_response(True, "Report returned successfully", report)


@router.delete("/{report_id}")
def remove_user(report_id: int, db: Session = Depends(get_db)):
    deleted = report_service.delete_report(db, report_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Report not found")
    return format_response(True, "Report deleted successfully")
