"""03-2 실습: 같은 Judge 요청을 세 가지 방식으로 보내고, 결과를 JSON으로 읽을 수 있는지 봅니다.

3번은 프롬프트에서 JSON 형식 지시를 빼고 스키마만 넘깁니다.
서버가 스키마를 정말 강제한다면 이때도 JSON이 나와야 합니다.

실행: python -m ch03.structured_output
"""
import json

from ch03.judges import BINARY_SYSTEM
from common.llm import JUDGE_MODEL, client

# 판정 기준만 남기고 "다음 JSON 형식으로만 답하세요" 이후를 뺀 프롬프트
CRITERIA_ONLY = BINARY_SYSTEM.split("\n\n다음 JSON")[0]

USER = "질문: 태양계에서 가장 큰 행성은 무엇인가요?\n정답: 태양계에서 가장 큰 행성은 목성입니다.\n답변: 태양계에서 가장 큰 행성은 토성입니다."

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "result": {"type": "string", "enum": ["PASS", "FAIL"]},
    },
    "required": ["reason", "result"],
    "additionalProperties": False,
}

MODES = {
    "1) 프롬프트로 JSON 요청": (BINARY_SYSTEM, {}),
    "2) 프롬프트 + json_object 모드": (BINARY_SYSTEM, {"response_format": {"type": "json_object"}}),
    "3) 형식 지시 없이 json_schema 모드만": (CRITERIA_ONLY, {"response_format": {
        "type": "json_schema", "json_schema": {"name": "judge", "schema": SCHEMA, "strict": True}}}),
}

for name, (system, options) in MODES.items():
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": USER}],
        **options,
    )
    text = response.choices[0].message.content
    try:
        parsed = json.loads(text)
        status = f"JSON 읽기 성공 → result={parsed.get('result')}"
    except json.JSONDecodeError:
        status = "JSON 읽기 실패"
    print(f"{name}\n  원문: {text}\n  {status}\n")
