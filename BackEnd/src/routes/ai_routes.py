from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
import io

# Importações locais da AI
from AI.model.model import modelo, classes
from AI.utils.transforms import tta_transforms
from AI.utils.tratamentos import tratamentos

router = APIRouter()

@router.post("/analyze")
async def classificar(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    all_probs = []
    with torch.no_grad():
        for t in tta_transforms:
            img_tensor = t(image).unsqueeze(0)
            saida = modelo(img_tensor)
            probs = F.softmax(saida, dim=1)
            all_probs.append(probs)

    probs_medias = torch.mean(torch.cat(all_probs, dim=0), dim=0)
    conf, pred = torch.max(probs_medias, 0)

    classe = classes[pred.item()]
    tratamento = tratamentos.get(classe, "Tratamento não encontrado.")

    return JSONResponse({
        "classe": classe,
        "confianca": float(conf.item()),
        "tratamento": tratamento
    })
