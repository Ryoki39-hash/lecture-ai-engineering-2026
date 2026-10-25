from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_token_usage_and_returns_answer(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="回答です",
    )
    request = {}

    def generate_content(**kwargs):
        request.update(kwargs)
        return response

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    times = iter([1.0, 1.25])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(times))

    answer = ask_module.ask("質問", model="gemini-test")

    assert answer == "回答です"
    assert request == {"model": "gemini-test", "contents": "質問"}
    output = capsys.readouterr().out
    assert "[応答時間] 0.25 秒" in output
    assert "入力: 12 / 出力: 34" in output


def test_main_runs_each_requested_model(monkeypatch, capsys):
    calls = []

    def fake_ask(question, model):
        calls.append((question, model))
        return f"{model}の回答"

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    monkeypatch.setattr(
        "sys.argv",
        [
            "app.ask",
            "--models",
            "gemini-first,gemini-second",
            "質問",
        ],
    )

    ask_module.main()

    assert calls == [("質問", "gemini-first"), ("質問", "gemini-second")]
    output = capsys.readouterr().out
    assert "[model] gemini-first" in output
    assert "[model] gemini-second" in output
    assert "gemini-firstの回答" in output
    assert "gemini-secondの回答" in output
