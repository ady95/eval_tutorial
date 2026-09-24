# eval_tutorial

위키독스 책 「LLM·RAG·에이전트 평가 따라하기 (LLM-as-a-Judge부터 RAGAS·DeepEval·Agent Eval까지)」의 예제 코드입니다.

- 책: [https://wikidocs.net/book/21419](https://wikidocs.net/book/21419)

## 시작하기

```bash
git clone https://github.com/ady95/eval_tutorial.git
cd eval_tutorial
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   macOS·Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # API 키와 모델 이름을 채웁니다
python -m ch01.check_env
```

실습 코드는 저장소 루트에서 `python -m <폴더>.<파일이름>` 형식으로 실행합니다.

## 구성

| 폴더 | 내용 |
|---|---|
| `common/` | 모든 장이 함께 쓰는 LLM 호출 도구 |
| `ch01/` | 01장 생성형 AI 평가는 무엇이 다른가 |
| `ch02/` | 02장 전통적인 지표와 그 한계 (추가 설치: `pip install sacrebleu rouge-score numpy`, 임베딩 서버 필요) |

장이 진행되면서 폴더가 추가됩니다.
