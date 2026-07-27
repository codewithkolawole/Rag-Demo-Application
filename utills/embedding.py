import numpy as np
from dotenv import load_dotenv
import os
from google import genai


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client()


def data_embed(doc):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=doc
    )
    return np.array(response.embeddings[0].values, dtype="float32")