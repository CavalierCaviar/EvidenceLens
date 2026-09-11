from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "EvidenceLens"
    secret_key: str = "dev-secret-change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7
    database_url: str = "sqlite:///./data/evidencelens.db"
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    data_dir: str = "./data"
    default_top_k: int = 12
    max_chunks_per_paper: int = 3
    candidate_similarity_threshold: float = 0.35
    contradiction_high: float = 0.85
    contradiction_moderate: float = 0.65

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "uploads").mkdir(exist_ok=True)
        (p / "models").mkdir(exist_ok=True)
        (p / "exports").mkdir(exist_ok=True)
        (p / "benchmarks").mkdir(exist_ok=True)
        return p


settings = Settings()
