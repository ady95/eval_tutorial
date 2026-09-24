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
| `ch03/` | 03장 LLM-as-a-Judge 직접 만들기 (PASS/FAIL, Rubric 점수, Pairwise Judge) |
| `ch04/` | 04장 Judge를 믿어도 되는가 (편향 실험, 사람 라벨 비교, 로컬 Judge) |
| `ch05/` | 05장 RAG 평가의 구조 (평가용 RAG, 검색 지표, Faithfulness) |
| `ch06/` | 06장 RAGAS로 RAG 평가하고 개선하기 (추가 설치: `pip install "ragas==0.4.3" "langchain-community==0.4.1"`, Reranker는 `torch`·`sentence-transformers`) |
| `ch07/` | 07장 DeepEval로 LLM 테스트하기 (추가 설치: `pip install "deepeval==4.2.5"`, Windows는 `pip install "portalocker[win32]"`도 필요. `.env`에 `DEEPEVAL_TELEMETRY_OPT_OUT=YES`) |
| `ch08/` | 08장 Agent 평가 (도구 3종 Agent와 Trace, 도구·경로·계획 지표, DeepEval Agent 지표) |
| `ch09/` | 09장 Evaluation Engineering (합성 평가셋, 회귀 평가와 배포 게이트, 실패 분류, 품질·비용·지연 비교). CI 예시는 `.github/workflows/llm-eval.yml` (수동 실행 전용) |
| `data/` | 05장부터 쓰는 공통 실습 데이터: 가상 회사 누리솔 사내 규정 23개(`docs/`), 평가셋 42문항(`eval_set.json`), Agent 평가 과제 20개(`agent_tasks.json`), 가상 날씨 예보(`weather.json`) |

장이 진행되면서 폴더가 추가됩니다.
