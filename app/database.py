import pandas as pd
from supabase import create_client
from .config import settings

class SupabaseConnector:
    def __init__(self):
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_KEY
        self.client = None
    
    def connect(self):
        try:
            self.client = create_client(self.url, self.key)
            return True
        except Exception as e:
            print(f"Supabase connection failed: {e}")
            return False
    
    def execute_query(self, query):
        try:
            if "SELECT * FROM " in query and "(" in query:
                func_name = query.split("SELECT * FROM ")[1].split("(")[0].strip()
                result = self.client.rpc(func_name).execute()
                return pd.DataFrame(result.data) if result.data else pd.DataFrame()
        except Exception as e:
            print(f"Query execution failed: {e}")
            return None
    
    def close(self):
        if self.client:
            self.client.auth.sign_out()