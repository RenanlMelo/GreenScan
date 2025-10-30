from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from PIL import Image
import torch
import torch.nn.functional as F
import io, os, uuid

from AI.model.model import modelo, classes
from AI.utils.transforms import tta_transforms
from AI.utils.responses import disease_responses
from src.database.connection import get_db
from src.database.models import Report

router = APIRouter()

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")
async def analyze_image(image: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1️⃣ Salva imagem localmente
    file_ext = image.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as f:
        f.write(await image.read())

    # 2️⃣ Processa a imagem com a IA
    img = Image.open(file_path).convert("RGB")

    all_probs = []
    with torch.no_grad():
        for t in tta_transforms:
            tensor = t(img).unsqueeze(0)
            output = modelo(tensor)
            probs = F.softmax(output, dim=1)
            all_probs.append(probs)

    probs_mean = torch.mean(torch.cat(all_probs, dim=0), dim=0)
    conf, pred = torch.max(probs_mean, 0)

    classe = classes[pred.item()]
    detalhes = disease_responses.get(classe, {})

    # 3️⃣ Salva no banco
    new_report = Report(
        clss=classe,
        trust=float(conf.item()),
        title=detalhes.get("title", "Title not found."),
        description=detalhes.get("description", "Description not found."),
        treatment=detalhes.get("treatment", "Treatment not found."),
        prevention=detalhes.get("prevention", "Prevention not found"),
        image=f"uploads/{file_name}"
    )

    db.add(new_report)
    db.commit()
    db.refresh(new_report)

    # 4️⃣ Monta URL da imagem
    image_url = f"/uploads/{file_name}"

    # 5️⃣ Retorna todos os dados já completos
    return JSONResponse({
        "success": True,
        "message": "Report created successfully",
        "data": {
            "id": new_report.id,
            "classe": classe,
            "confidence": float(conf.item()),
            "detalhes": detalhes,
            "image_url": image_url
        }
    })
