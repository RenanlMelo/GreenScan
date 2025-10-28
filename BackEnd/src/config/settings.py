from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    PORT = int(os.getenv("PORT", 8000))
    DB_URL = os.getenv("DB_URL", "sqlite:///./database.db")

settings = Settings()
