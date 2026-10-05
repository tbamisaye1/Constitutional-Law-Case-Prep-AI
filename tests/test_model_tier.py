from app.llm.openrouter import normalize_model_tier, resolve_chat_model_id


def test_normalize_model_tier():
    assert normalize_model_tier(None) == "standard"
    assert normalize_model_tier("standard") == "standard"
    assert normalize_model_tier("ADVANCED") == "advanced"
    assert normalize_model_tier("weird") == "standard"


def test_resolve_prefers_openai_for_advanced(monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "openrouter_model", "openai/gpt-4o-mini")
    monkeypatch.setattr(settings, "openrouter_advanced_model", "openai/gpt-5-mini")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_advanced_web_model", "gpt-5-mini")
    monkeypatch.setattr(settings, "openai_web_model", "gpt-5-mini")

    assert resolve_chat_model_id("standard") == "openai/gpt-4o-mini"
    assert resolve_chat_model_id("advanced") == "gpt-5-mini"


def test_resolve_falls_back_to_openrouter_without_openai(monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "openrouter_model", "openai/gpt-4o-mini")
    monkeypatch.setattr(settings, "openrouter_advanced_model", "openai/gpt-5-mini")
    monkeypatch.setattr(settings, "openai_api_key", "")

    assert resolve_chat_model_id("advanced") == "openai/gpt-5-mini"
