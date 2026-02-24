from openai import OpenAI
import os
import supabase


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Get the embedding of a text using the OpenAI APU

    Args:
        client (OpenAI): OpenAI client with API key
        text (str): Text to get the embedding of

    Returns:
        list[float]: Embedding of the text
    """
    embedding = (
        client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )
    return embedding


def get_profile(profile_id: int):
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    client = supabase.create_client(url, key)
    res = (
        client.table("volunteer_profiles")
        .select("*")
        .eq("user_id", profile_id)
        .execute()
    )
    return res.data[0]


def get_opportunities():
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    client = supabase.create_client(url, key)
    res = client.table("opportunities").select("*").execute()
    return res.data
