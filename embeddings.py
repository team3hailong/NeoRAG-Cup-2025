from openai import OpenAI
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

# Các em có thể tự thêm embedding model mới hoặc dùng các model có sẵn
class Embeddings:
    def __init__(self, model_name, type):
        self.model_name = model_name
        self.type = type
        if type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif type == "ollama":
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
        elif type == "sentence_transformers":
            self.client = SentenceTransformer(model_name, trust_remote_code=True)
        elif type == "gemini":
            self.client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )

    def encode(self, doc):
        if self.type in ["openai", "ollama"]:
            return self.client.embeddings.create(
                input=doc,
                model=self.model_name
            ).data[0].embedding
        elif self.type == "sentence_transformers":
            embedding = self.client.encode(doc)
            return embedding.tolist()
        elif self.type == "gemini":
            return self.client.models.embed_content(
                model=self.model_name,
                contents=doc
            ).embeddings[0].values