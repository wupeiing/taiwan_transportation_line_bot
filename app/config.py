from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    line_channel_secret: str
    line_channel_access_token: str
    groq_api_key: str
    tdx_client_id: str
    tdx_client_secret: str

    model_config = {"env_file": ".env"}


settings = Settings()
