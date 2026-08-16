"""Application configuration, loaded from environment variables or a .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for every runtime setting.

    Values are read from the process environment first, then from a local
    .env file. Nothing in this project should read os.environ directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    llm_temperature: float = 0.7
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3

    state_dir: str = "states"
    log_level: str = "INFO"
    log_to_file: bool = True

    @property
    def is_configured(self) -> bool:
        """True when an API key is present, so the UI can warn before running."""
        return bool(self.groq_api_key.strip())


def get_settings() -> Settings:
    """Build a Settings instance. Call this instead of instantiating directly."""
    return Settings()
