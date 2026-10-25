from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_token_usage_and_returns_answer(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="回答です",
    )
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)

    answer = ask_module.ask("質問")

    assert answer == "回答です"
    assert "入力: 12 / 出力: 34" in capsys.readouterr().out