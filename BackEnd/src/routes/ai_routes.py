from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from PIL import Image
import torch
import torch.nn.functional as F
import io, os, uuid, time

from AI.model.model import modelo, classes
from AI.utils.transforms import tta_transforms
from AI.utils.responses import disease_responses
from src.database.connection import get_db
from src.database.models import Report

router = APIRouter()

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def analyze_with_retry(img, max_retries=3, delay=1):
    """Processa a imagem com o modelo e tenta novamente em caso de erro."""
    for attempt in range(1, max_retries + 1):
        try:
            all_probs = []
            with torch.no_grad():
                for t in tta_transforms:
                    tensor = t(img).unsqueeze(0)
                    output = modelo(tensor)
                    probs = F.softmax(output, dim=1)
                    all_probs.append(probs)

            probs_mean = torch.mean(torch.cat(all_probs, dim=0), dim=0)
            conf, pred = torch.max(probs_mean, 0)
            return conf, pred
        except Exception as e:
            print(f"⚠️ Erro ao processar imagem (tentativa {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                raise e
            time.sleep(delay)


@router.post("/analyze")
async def analyze_image(image: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # 1️⃣ Salva imagem localmente
        file_ext = image.filename.split(".")[-1]
        file_name = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, file_name)

        with open(file_path, "wb") as f:
            f.write(await image.read())

        # 2️⃣ Processa a imagem com retry
        img = Image.open(file_path).convert("RGB")
        
        conf, pred = analyze_with_retry(img)

        # 3️⃣ Interpreta resultados
        classe = classes[pred.item()]
        detalhes = disease_responses.get(classe, {})

        # 4️⃣ Salva no banco
        new_report = Report(
            clss=classe,
            trust=float(conf.item()),
            title=detalhes.get("title", "Title not found."),
            description=detalhes.get("description", "Description not found."),
            treatment=detalhes.get("treatment", "Treatment not found."),
            prevention=detalhes.get("prevention", "Prevention not found."),
            image=f"uploads/{file_name}"
        )

        db.add(new_report)
        db.commit()
        db.refresh(new_report)

        # 5️⃣ Monta resposta
        image_url = f"/uploads/{file_name}"

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

    except Exception as e:
        print(f"❌ Erro ao analisar imagem: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Erro ao processar imagem: {e}"}
        )
