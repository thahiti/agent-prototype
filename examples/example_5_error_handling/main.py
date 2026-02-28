"""
예제 5: 표준화된 에러 핸들링 (Fallback & Auto-correction)

핵심 아이디어:
- LLM은 불안정하다: JSON 형식을 틀리거나, 필수 필드를 누락하거나, API가 일시 장애를 일으킨다.
- 두 가지 레벨의 에러 핸들링을 구현한다:
  1) 시스템 레벨: retry_policy로 네트워크 에러 등 일시 장애를 자동 재시도
  2) 로직 레벨: LLM이 잘못된 형식을 출력하면 에러 메시지와 함께 다시 시도하게 하는 자가 교정 루프

실행 방법:
    uv run examples/example_5_error_handling/main.py

실행 흐름:
    [START] → [llm_node] → (라우터) → [END]         (성공 시)
                              ↘ [correction_node] → [llm_node]  (파싱 실패 시, 루프)
"""

import json
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1단계: State 정의 — 에러 관련 필드 포함
# ============================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # 파싱 에러 메시지 (없으면 빈 문자열)
    parse_error: str
    # 자가 교정 재시도 횟수
    retry_count: int
    # 최종 파싱된 결과
    parsed_result: dict


# ============================================================
# 2단계: LLM 노드 — JSON 형식 응답을 요청
# ============================================================
def llm_node(state: State) -> dict:
    """LLM에게 JSON 형식의 응답을 요청하는 노드.

    이 노드가 핵심: LLM이 JSON을 틀릴 수 있기 때문에
    파싱을 시도하고, 실패하면 에러 정보를 State에 기록한다.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 이전에 파싱 에러가 있었다면, 교정 지시를 메시지에 포함
    if state.get("parse_error"):
        print(f"[LLM] 이전 에러가 있음 → 교정 모드 (시도 #{state['retry_count']})")

    response = llm.invoke(state["messages"])
    content = response.content

    print(f"[LLM] 응답: {content[:100]}...")

    # --- JSON 파싱 시도 ---
    try:
        # LLM 응답에서 JSON 블록 추출
        parsed = extract_json(content)
        print(f"[LLM] JSON 파싱 성공: {parsed}")

        return {
            "messages": [response],
            "parse_error": "",
            "parsed_result": parsed,
        }

    except (json.JSONDecodeError, ValueError) as e:
        # 파싱 실패 → 에러 정보를 State에 기록
        error_msg = f"JSON 파싱 실패: {str(e)}"
        print(f"[LLM] ❌ {error_msg}")

        return {
            "messages": [response],
            "parse_error": error_msg,
            "retry_count": state.get("retry_count", 0) + 1,
            "parsed_result": {},
        }


def extract_json(text: str) -> dict:
    """LLM 응답에서 JSON을 추출한다.
    ```json ... ``` 블록이 있으면 그 안의 내용을, 없으면 전체를 파싱한다.
    """
    # ```json ... ``` 블록 추출 시도
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + len("```")
        end = text.index("```", start)
        text = text[start:end].strip()

    return json.loads(text)


# ============================================================
# 3단계: 라우터 — 파싱 성공/실패에 따라 분기
# ============================================================
MAX_RETRIES = 3  # 최대 재시도 횟수


def error_router(state: State) -> str:
    """파싱 에러 여부와 재시도 횟수에 따라 다음 노드를 결정한다.

    - 에러 없음 → 종료
    - 에러 있고 재시도 가능 → 교정 노드로
    - 에러 있고 재시도 초과 → 강제 종료 (무한 루프 방지)
    """
    if not state.get("parse_error"):
        print("[ROUTER] 파싱 성공 → END")
        return END

    if state.get("retry_count", 0) >= MAX_RETRIES:
        print(f"[ROUTER] 재시도 한도({MAX_RETRIES}회) 초과 → END (실패)")
        return END

    print(f"[ROUTER] 파싱 실패 → 교정 노드로 이동")
    return "correction"


# ============================================================
# 4단계: 자가 교정 노드 — 에러 메시지를 LLM에게 피드백
# ============================================================
def correction_node(state: State) -> dict:
    """파싱 에러 메시지를 새로운 HumanMessage로 감싸서
    LLM에게 다시 시도하도록 요청하는 노드.

    핵심 기법: 에러 메시지 자체를 LLM의 입력으로 넣어서
    LLM이 스스로 실수를 인지하고 교정하게 만든다.
    """
    error = state["parse_error"]
    retry = state.get("retry_count", 0)

    # 교정 지시 메시지 구성
    correction_prompt = (
        f"이전 응답에서 형식 오류가 발생했습니다.\n"
        f"에러 내용: {error}\n\n"
        f"반드시 유효한 JSON 형식으로만 응답하세요.\n"
        f"다른 텍스트 없이 JSON만 출력하세요.\n"
        f"(재시도 {retry}/{MAX_RETRIES})"
    )

    print(f"[CORRECTION] 교정 프롬프트 생성 (시도 {retry}/{MAX_RETRIES})")

    return {
        "messages": [HumanMessage(content=correction_prompt)],
        "parse_error": "",  # 에러 초기화 → 다시 LLM 노드에서 파싱 시도
    }


# ============================================================
# 5단계: 그래프 조립 — 자가 교정 루프 포함
# ============================================================
def build_graph():
    """핵심 구조: llm_node → (에러시) correction_node → llm_node (루프)

    retry_policy는 시스템 레벨 재시도(네트워크 에러 등)를 처리하고,
    correction 루프는 로직 레벨 재시도(형식 에러 등)를 처리한다.
    """
    graph = StateGraph(State)

    # 노드 등록
    graph.add_node("llm", llm_node)
    graph.add_node("correction", correction_node)

    # 엣지 연결
    graph.add_edge(START, "llm")
    # LLM 노드 → 라우터(조건부 분기)
    graph.add_conditional_edges("llm", error_router, ["correction", END])
    # 교정 노드 → 다시 LLM 노드 (루프)
    graph.add_edge("correction", "llm")

    return graph.compile()


# ============================================================
# 6단계: 데모 실행
# ============================================================
def main():
    app = build_graph()

    print("=" * 60)
    print("예제 5: 에러 핸들링 (자가 교정 루프) 데모")
    print("=" * 60)

    # JSON 형식 응답을 요청하는 프롬프트
    result = app.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "대한민국의 수도, 인구, 공식 언어를 알려주세요.\n"
                    "반드시 아래 JSON 형식으로만 응답하세요:\n"
                    '{"capital": "...", "population": "...", "language": "..."}'
                )
            )
        ],
        "parse_error": "",
        "retry_count": 0,
        "parsed_result": {},
    })

    print(f"\n--- 최종 결과 ---")
    print(f"파싱된 데이터: {result['parsed_result']}")
    print(f"총 재시도 횟수: {result['retry_count']}")
    if result["parse_error"]:
        print(f"최종 에러: {result['parse_error']}")
    else:
        print("상태: 성공 ✅")


if __name__ == "__main__":
    main()
