"""08장 Agent 가 쓰는 도구 3종: 사내 문서 검색, 계산기, 날씨 조회.

도구 함수는 항상 문자열을 돌려줍니다. 오류도 예외 대신 "오류: ..." 문자열로 돌려주어,
Agent 가 결과를 보고 다음 행동을 고를 수 있게 합니다.
"""
import ast
import json
import operator
from pathlib import Path

from ch05.rag import Retriever

_retriever = None
WEATHER = json.loads(Path("data/weather.json").read_text(encoding="utf-8"))["예보"]


def search_document(query: str) -> str:
    """05장의 검색기로 누리솔 사내 규정에서 관련 청크 3개를 찾습니다."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(400)  # 처음 부를 때 한 번만 색인을 만듭니다
    hits = _retriever.search(query, 3)
    return "\n\n---\n\n".join(chunk.text for chunk, _ in hits)


_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.USub: operator.neg}


def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("숫자와 + - * / 괄호만 쓸 수 있습니다")


def calculator(expression: str) -> str:
    """사칙연산 식을 계산합니다. eval 을 쓰지 않고 식의 구조를 직접 읽습니다."""
    try:
        value = _evaluate(ast.parse(expression.replace(",", ""), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return f"오류: {e}"
    return str(int(value)) if float(value).is_integer() else str(round(value, 4))


def get_weather(city: str, date: str) -> str:
    """가상 날씨 예보(data/weather.json)에서 도시와 날짜(YYYY-MM-DD)의 예보를 찾습니다."""
    if city not in WEATHER:
        return f"오류: '{city}'은(는) 조회할 수 없는 도시입니다. 가능한 도시: {', '.join(WEATHER)}"
    if date not in WEATHER[city]:
        return f"오류: {date} 예보가 없습니다. 조회 가능한 날짜: {min(WEATHER[city])} ~ {max(WEATHER[city])}"
    return json.dumps({"도시": city, "날짜": date, **WEATHER[city][date]}, ensure_ascii=False)


TOOLS = {"search_document": search_document, "calculator": calculator, "get_weather": get_weather}

# LLM 에게 알려 줄 도구 설명 (OpenAI tools 형식)
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_document",
        "description": "누리솔 사내 규정 문서에서 질문과 관련된 부분을 찾습니다.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "찾을 내용을 담은 검색어"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "사칙연산 식을 계산합니다. 예: 120000*2+30000*3",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "숫자와 + - * / 괄호로 된 식"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "도시의 날짜별 날씨 예보를 조회합니다.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "도시 이름. 예: 서울"},
            "date": {"type": "string", "description": "날짜. YYYY-MM-DD 형식"}}, "required": ["city", "date"]}}},
]

# 처음 만든 버전(v0)의 도구 설명: 이름과 인자는 같고 설명만 짧습니다 (08-5 에서 위의 설명으로 고칩니다)
TOOL_SCHEMAS_V0 = [
    {"type": "function", "function": {
        "name": "search_document", "description": "문서를 검색합니다.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "계산을 합니다.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "날씨를 조회합니다.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"}},
                       "required": ["city", "date"]}}},
]
