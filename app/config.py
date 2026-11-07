from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    cohere_api_key: str
    chroma_path: str = "./chroma_db"
    database_url: str

    # Ingestion
    raw_data_dir: str = "data/raw"
    collection_name: str = "health_docs"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embed_batch_size: int = 100

    # Generation
    llm_model: str = "gpt-4o-mini"
    prompt_path: str = "prompts/rag_prompt.yaml"

    # Hybrid retrieval
    candidate_k: int = 20
    rerank_model: str = "rerank-v3.5"

    # Evaluation (eval/run_eval.py)
    eval_judge_model: str = "gpt-4o-mini"
    eval_top_k: int = 5
    eval_rerank_delay_s: float = 6.5  # trial Cohere keys allow ~10 rerank calls/min

    class Config:
        env_file = ".env"


settings = Settings()
