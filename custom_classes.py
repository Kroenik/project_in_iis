from dataclasses import dataclass
from supabase import Client


@dataclass
class Context:
    user_id: int
    supabase: Client
