import chromadb
from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)
chroma = chromadb.PersistentClient(path=settings.chroma_path)
collection = chroma.get_or_create_collection('health_docs')

print(f'Total chunks: {collection.count()}')