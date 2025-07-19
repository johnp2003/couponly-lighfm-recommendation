import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SUPABASE_URL = os.getenv("EXPO_PUBLIC_SUPABASE_URL")
    SUPABASE_KEY = os.getenv("EXPO_PUBLIC_SUPABASE_ANON_KEY")

settings = Settings()