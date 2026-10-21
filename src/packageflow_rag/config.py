"""Settings loaded from .env (see .env.example).

Every setting has a default, so a missing line in .env is fine.
Run `python -m src.packageflow_rag.config` from the project root to print them.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_num_ctx: int = 8192
    ollama_timeout: float = 300.0          # seconds; first call loads the model and can be slow
    llm_temperature: float = 0.1
    embed_model: str = "qwen3-embedding:0.6b"
    embed_batch_size: int = 16
    embed_num_ctx: int = 8192              # longest chunk the embedder will read before truncating
    vision_model: str = "qwen2.5vl:7b"     # only for scripts/describe_image.py

    # Paths
    source_dir: Path = Path("data/source")
    derived_dir: Path = Path("data/derived")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("~/.packageflow-rag/index")
    log_dir: Path = Path("~/.packageflow-rag/logs")
    collection_name: str = "packageflow_docs"

    # Chunking
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # Retrieval
    top_k: int = 8
    candidate_k: int = 25                  # how many hits each of dense / BM25 contributes before fusion
    rrf_k: int = 60
    min_dense_score: float = 0.0           # 0 = off. If the best cosine score is below this, answer "not covered"

    # Generation
    answer_reserve_tokens: int = 1024      # room left in num_ctx for the answer + prompt text

    def resolve(self, p: Path) -> Path:
        """Expand ~ and make relative paths relative to the project root."""
        p = Path(p).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p


settings = Settings()

if __name__ == "__main__":
    for k, v in settings.model_dump().items():
        print(f"{k:22} {v}")
