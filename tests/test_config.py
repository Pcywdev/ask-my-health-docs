from app.config import settings


def test_settings_use_dummy_values() -> None:
    """Guard: the suite must never run with real keys or a real database."""
    assert settings.openai_api_key == "test-openai-key"
    assert settings.cohere_api_key == "test-cohere-key"
    assert settings.database_url == "sqlite://"
