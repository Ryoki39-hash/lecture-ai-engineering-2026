"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --models gemini-3.5-flash-lite,gemini-3.1-flash-lite "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, model: str = MODEL) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    response = client.models.generate_content(model=model, contents=question)
    elapsed = time.perf_counter() - started_at
    #responseの間にあるので応答時間を測る．

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    print(f"[応答時間] {elapsed:.2f} 秒")
    print(
        f"[トークン数] 入力: {input_tokens if input_tokens is not None else '不明'}"
        f" / 出力: {output_tokens if output_tokens is not None else '不明'}"
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument(
        "--models",
        metavar="MODEL",
        help="比較するモデル名（カンマ区切り、省略時は AIE_MODEL または既定モデル）",
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()
    models = [model.strip() for model in args.models.split(",")] if args.models else [MODEL]
    if any(not model for model in models):
        parser.error("--models には空でないモデル名を指定してください")

    print(f"[質問] {args.question}\n")
    failed = False
    for model in models:
        print(f"[model] {model}")
        try:
            print(ask(args.question, model=model))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            failed = True
        print()

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
