from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    cohere_api_key: str = ""
    chroma_path: str = "./chroma_db"

    # Ingestion
    raw_data_dir: str = "data/raw"
    collection_name: str = "health_docs"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embed_batch_size: int = 100

    class Config:
        env_file = ".env"


settings = Settings()
