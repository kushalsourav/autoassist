from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')

    storage_mode: str = 'local'
    model_variant: str = 'baseline'  # baseline, distilbert, or multilabel_distilbert_v3
    database_path: str = 'backend/autoassist.db'
    supabase_url: str = ''
    supabase_anon_key: str = ''
    openrouter_api_key: str = ''
    openrouter_model: str = ''
    openrouter_timeout_seconds: float = Field(default=90, ge=5, le=180)
    # Anthropic (default provider)
    anthropic_api_key: str = ''
    anthropic_model: str = 'claude-haiku-5-5'
    anthropic_max_tokens: int = Field(default=4096, ge=256, le=32768)
    anthropic_timeout_seconds: float = Field(default=90, ge=5, le=180)
    # Active provider: 'anthropic' (default) | 'openrouter'
    autoassist_provider: str = 'anthropic'
    cors_origins: str = 'http://localhost:3000'
    rag_backend: str = 'tfidf'  # Set to faiss after installing optional RAG dependencies
    rag_min_similarity: float = Field(default=0.35, ge=0, le=1)
    rag_corpus_mode: str = 'off'  # off | shadow | prefer
    rag_corpus_dir: str = 'ml/artifacts/recall_corpus_v2'

    @property
    def database_file(self) -> Path:
        path = Path(self.database_path)
        return path if path.is_absolute() else ROOT / path

    @property
    def transformer_dir(self) -> Path:
        return ROOT / 'ml' / 'artifacts' / 'distilbert'

    @property
    def multilabel_v3_dir(self) -> Path:
        return ROOT / 'ml' / 'artifacts' / 'multilabel_distilbert_v3'

    @property
    def model_file(self) -> Path:
        return ROOT / 'ml' / 'artifacts' / 'baseline.joblib'


settings = Settings()
