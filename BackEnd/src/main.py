from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.middleware.logger import logger
from src.routes.routes import router
from src.database.connection import Base, engine, connect_database
from src.config.settings import settings

# Start running
# uvicorn src.main:app --host 0.0.0.0 --reload --port 8000

app = FastAPI(title="Projeto Backend com FastAPI + SQLAlchemy")
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware
app.middleware("http")(logger)

# Rotas
app.include_router(router)

# Cria tabelas
Base.metadata.create_all(bind=engine)

@app.on_event("startup")
def startup_event():
    connect_database()
    print(f"🚀 Servidor rodando na porta {settings.PORT}")

@app.get("/")
def read_root():
    return {"message": "API funcionando corretamente 🚀"}
