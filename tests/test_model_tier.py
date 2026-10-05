from app.llm.openrouter import normalize_model_tier, resolve_chat_model_id


def test_normalize_model_tier():
    assert normalize_model_tier(None) == "standard"
    assert normalize_model_tier("standard") == "standard"
    assert normalize_model_tier("ADVANCED") == "advanced"
    assert normalize_model_tier("weird") == "standard"


def test_resolve_chat_model_id_defaults(monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "openrouter_model", "openai/gpt-4o-mini")
    monkeypatch.setattr(settings, "openrouter_advanced_model", "openai/gpt-5-mini")

    assert resolve_chat_model_id("standard") == "openai/gpt-4o-mini"
    assert resolve_chat_model_id("advanced") == "openai/gpt-5-mini"
