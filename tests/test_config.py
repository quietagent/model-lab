import os

from app import config


def unset(monkeypatch, name):
    """Remove a variable now, and guarantee it is gone again when the test ends."""
    monkeypatch.setenv(name, "")
    monkeypatch.delenv(name)


def test_env_file_is_loaded_without_overriding_real_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "OPENROUTER_API_KEY = sk-or-from-file\n"
        'HF_TOKEN="hf_quoted"\n'
        "export MODELLAB_TEST_EXTRA=1\n"
        "not a valid line\n",
        encoding="utf-8")
    unset(monkeypatch, "OPENROUTER_API_KEY")
    unset(monkeypatch, "MODELLAB_TEST_EXTRA")
    monkeypatch.setenv("HF_TOKEN", "hf_from_real_env")

    config.load_env_file(env)

    assert config.env_key("openrouter") == "sk-or-from-file"
    assert config.env_key("huggingface") == "hf_from_real_env"      # real env wins
    assert os.environ["MODELLAB_TEST_EXTRA"] == "1"


def test_missing_env_file_is_fine(tmp_path):
    config.load_env_file(tmp_path / "nope.env")


def test_mock_mode_uses_its_own_database(monkeypatch):
    monkeypatch.setenv("MOCK", "1")
    assert config.db_path().name == "modellab.mock.db"
    monkeypatch.setenv("MOCK", "0")
    assert config.db_path().name == "modellab.db"


def test_provider_endpoints():
    assert config.PROVIDERS["openrouter"]["chat_url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert config.PROVIDERS["huggingface"]["chat_url"] == "https://router.huggingface.co/v1/chat/completions"
    assert config.PROVIDERS["openrouter"]["env"] == "OPENROUTER_API_KEY"
    assert config.PROVIDERS["huggingface"]["env"] == "HF_TOKEN"
