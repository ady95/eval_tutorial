"""01-2 실습 환경 점검: 설정한 모델로 질문 하나를 보내 봅니다.

실행: python -m ch01.check_env
"""
from common.llm import JUDGE_MODEL, MODEL, client

print(f"답변 모델 : {MODEL}")
print(f"Judge 모델: {JUDGE_MODEL}")

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "대한민국의 수도는 어디인가요? 한 문장으로 답하세요."}],
)
print(f"답변      : {response.choices[0].message.content}")
print(f"토큰 사용 : 입력 {response.usage.prompt_tokens} / 출력 {response.usage.completion_tokens}")
