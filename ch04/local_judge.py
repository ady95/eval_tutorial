"""04-4 로컬 Judge 설정: 명령행 옵션으로 Ollama 모델을 Judge로 바꿔 끼웁니다."""
import argparse

from openai import OpenAI

from ch03 import judges


def add_local_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local", action="store_true", help="Ollama 로컬 모델을 Judge로 사용")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--think", action="store_true", help="로컬 모델의 추론 모드를 켭니다")


def apply_local(args: argparse.Namespace) -> str:
    """--local 이면 Judge를 로컬 모델로 바꾸고, 결과 파일 이름에 쓸 Judge 이름을 돌려줍니다."""
    if args.local:
        options = {} if args.think else {"reasoning_effort": "none"}  # 추론 모드를 끄면 훨씬 빠릅니다
        judges.use_judge(OpenAI(base_url=args.base_url, api_key="ollama"), args.model, **options)
    name = judges.JUDGE["model"] + ("-think" if args.local and args.think else "")
    return name.replace(":", "_")
