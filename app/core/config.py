from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/xiaosui"
    DASHSCOPE_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    QWEN_API_KEY: str = ""

    # 阿里云 DashScope
    DASHSCOPE_STT_MODEL: str = "paraformer-v2"
    DASHSCOPE_EMBEDDING_MODEL: str = "text-embedding-v3"

    # LLM
    LLM_MODEL: str = "deepseek-chat"
    LLM_BASE_URL: str = "https://api.deepseek.com"

    class Config:
        env_file = ".env"


settings = Settings()
