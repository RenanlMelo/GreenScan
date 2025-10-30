# src/main.py
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.middleware.logger import logger
from src.routes.routes import router  # seu APIRouter central
from src.database.connection import Base, engine, connect_database
from src.config.settings import settings

# uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Caminho da pasta de uploads
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Cria app
app = FastAPI(title="Projeto Backend com FastAPI + SQLAlchemy")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, especifique os domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger assíncrono
async def async_logger(request, call_next):
    print(f"➡️  {request.method} {request.url}")
    response = await call_next(request)
    print(f"⬅️  Status: {response.status_code}")
    return response

app.middleware("http")(async_logger)

# Rotas
app.include_router(router)

# Serve arquivos estáticos
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Evento de startup
@app.on_event("startup")
def startup_event():
    # Conecta ao banco
    connect_database()
    # Cria tabelas se não existirem
    Base.metadata.create_all(bind=engine)
    print(f"🚀 Servidor rodando na porta {settings.PORT}")

# Endpoint raiz
@app.get("/")
def read_root():
    return {"message": "API funcionando corretamente 🚀"}
