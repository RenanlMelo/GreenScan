from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenetv2, MobileNet_V2_Weights
from PIL import Image
import io
import os

# === 1. Definição da arquitetura usada no treino ===
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Base pré-treinada
        self.model = mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # remove classifier final
        for param in self.model.parameters():
            param.requires_grad = False  # congela camadas base

        # Camadas adicionais (iguais às do seu treino)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# === 2. Carregar o modelo salvo (apenas pesos do state_dict) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r".\cnn_model.pt"

modelo = CNN()
modelo.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
modelo.eval()


# === 3. Transforms (TTA básico) ===
tta_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
]


# === 4. Classes ===
classes = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]


# === 5. Dicionário de tratamentos ===
tratamentos = {
    "Tomato___Bacterial_spot": "Aplicar fungicida à base de cobre.",
    "Tomato___Early_blight": "Retirar folhas infectadas e aplicar fungicida.",
    "Tomato___Late_blight": "Usar fungicida sistêmico, evitar irrigação por aspersão.",
    "Tomato___Leaf_Mold": "Melhorar ventilação da estufa e aplicar fungicida.",
    "Tomato___Septoria_leaf_spot": "Remover folhas afetadas e aplicar fungicida específico.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Aplicar acaricida e aumentar umidade.",
    "Tomato___Target_Spot": "Rotação de culturas e fungicidas protetores.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Controlar mosca-branca (vetor).",
    "Tomato___Tomato_mosaic_virus": "Evitar contato entre plantas, usar sementes certificadas.",
    "Tomato___healthy": "A planta está saudável, sem necessidade de tratamento."
}


# === 6. Inicializar FastAPI ===
app = FastAPI()


# === 7. Endpoint de classificação ===
@app.post("/classificar/")
async def classificar(file: UploadFile = File(...)):
    # Ler bytes da imagem enviada pelo app
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Aplicar TTA (Test Time Augmentation)
    all_probs = []
    with torch.no_grad():
        for t in tta_transforms:
            img_tensor = t(image).unsqueeze(0)
            saida = modelo(img_tensor)
            probs = F.softmax(saida, dim=1)
            all_probs.append(probs)

    # Média das probabilidades entre as variações
    probs_medias = torch.mean(torch.cat(all_probs, dim=0), dim=0)
    conf, pred = torch.max(probs_medias, 0)

    classe = classes[pred.item()]
    tratamento = tratamentos.get(classe, "Tratamento não encontrado.")

    # Retornar JSON com resultado
    return JSONResponse({
        "classe": classe,
        "confianca": float(conf.item()),
        "tratamento": tratamento
    })
