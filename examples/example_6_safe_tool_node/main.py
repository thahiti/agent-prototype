"""
예제 6: 도구 실행기 표준화 (SafeToolNode)

핵심 아이디어:
- 도구(Tool) 실행 중 예외가 발생하면 전체 에이전트가 다운되는 것을 방지한다.
- 기본 ToolNode를 래핑한 SafeToolNode가 예외를 캡슐화하여
  ToolMessage(status="error")로 변환한다.
- 에이전트는 에러 메시지를 읽고, 다른 도구를 시도하거나
  사용자에게 실패를 보고하는 등 논리적으로 대응할 수 있다.

실행 방법:
    uv run examples/example_6_safe_tool_node/main.py

실행 흐름:
    [START] → [agent] → (tool_call 있음?) → [safe_tools] → [agent] → ... → [END]
"""

import json
import random
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# 1단계: State 정의
# ============================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 2단계: 도구 정의 — 정상 도구 + 실패하는 도구
# ============================================================
@tool
def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 조회한다. (항상 성공)"""
    # 시뮬레이션: 실제로는 외부 API를 호출
    weathers = ["맑음 ☀️ 22°C", "흐림 ☁️ 18°C", "비 🌧️ 15°C"]
    result = random.choice(weathers)
    return f"{city}의 현재 날씨: {result}"


@tool
def search_database(query: str) -> str:
    """데이터베이스를 검색한다. (항상 실패 — 에러 시연용)"""
    # 의도적으로 예외를 발생시켜 SafeToolNode의 동작을 시연
    raise ConnectionError(
        f"데이터베이스 연결 실패: 'main-db' 서버에 접근할 수 없습니다. "
        f"(쿼리: '{query}')"
    )


@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산한다. (성공)"""
    result = eval(expression)  # 데모용 단순 계산
    return f"계산 결과: {expression} = {result}"


# 사용할 도구 목록
tools = [get_weather, search_database, calculate]


# ============================================================
# 3단계: SafeToolNode 구현 — 핵심 클래스
# ============================================================
class SafeToolNode:
    """ToolNode를 래핑하여 도구 실행 중 예외를 안전하게 처리한다.

    일반 ToolNode는 도구에서 예외가 발생하면 전체 그래프가 중단된다.
    SafeToolNode는 예외를 ToolMessage(status="error")로 캡슐화하여
    에이전트가 에러를 인지하고 논리적으로 대응할 수 있게 만든다.
    """

    def __init__(self, tools: list):
        """도구 목록을 받아 내부 ToolNode를 생성한다."""
        self.tool_node = ToolNode(tools)
        # 도구 이름 → 도구 객체 매핑 (직접 호출용)
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, state: State) -> dict:
        """도구를 실행하되, 예외가 발생하면 에러 메시지로 캡슐화한다.

        동작 원리:
        1. 마지막 AI 메시지에서 tool_calls를 추출한다.
        2. 각 tool_call을 개별적으로 실행한다.
        3. 성공하면 정상 ToolMessage를, 실패하면 에러 ToolMessage를 반환한다.
        4. 하나의 도구가 실패해도 다른 도구는 정상 실행된다.
        """
        messages = state["messages"]
        last_message = messages[-1]

        # AI 메시지에서 tool_calls 추출
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return {"messages": []}

        result_messages = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print(f"\n[SAFE_TOOL] 도구 실행: {tool_name}({tool_args})")

            try:
                # --- 정상 실행 경로 ---
                tool_fn = self.tools_by_name[tool_name]
                result = tool_fn.invoke(tool_args)

                print(f"[SAFE_TOOL] ✅ 성공: {result[:80]}...")

                result_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

            except Exception as e:
                # --- 에러 캡슐화 경로 ---
                # 예외를 ToolMessage로 변환 → 그래프가 계속 실행됨
                error_content = (
                    f"[도구 실행 에러]\n"
                    f"도구: {tool_name}\n"
                    f"에러 유형: {type(e).__name__}\n"
                    f"에러 내용: {str(e)}\n"
                    f"대안을 시도하거나 사용자에게 알려주세요."
                )

                print(f"[SAFE_TOOL] ❌ 에러 캡슐화: {type(e).__name__}: {e}")

                result_messages.append(
                    ToolMessage(
                        content=error_content,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        # status="error" 를 추가하면 에이전트가 에러임을 더 명확히 인식
                    )
                )

        return {"messages": result_messages}


# ============================================================
# 4단계: 에이전트 노드 — 도구를 사용하는 LLM
# ============================================================
def agent_node(state: State) -> dict:
    """도구가 바인딩된 LLM을 호출하는 에이전트 노드."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # LLM에 도구 목록을 바인딩 → LLM이 tool_call을 생성할 수 있게 됨
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ============================================================
# 5단계: 라우터 — tool_call 유무에 따라 분기
# ============================================================
def should_use_tools(state: State) -> str:
    """마지막 메시지에 tool_calls가 있으면 도구 노드로,
    없으면 종료한다.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# ============================================================
# 6단계: 그래프 조립 — 에이전트 ↔ SafeToolNode 루프
# ============================================================
def build_graph():
    """에이전트 ↔ SafeToolNode 루프 구조.

    에이전트가 tool_call을 생성하면 SafeToolNode가 실행하고,
    결과를 다시 에이전트에게 전달한다.
    SafeToolNode 덕분에 도구 에러가 발생해도 그래프가 중단되지 않는다.
    """
    graph = StateGraph(State)

    # SafeToolNode 인스턴스 생성
    safe_tools = SafeToolNode(tools)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", safe_tools)  # SafeToolNode를 노드로 등록

    graph.add_edge(START, "agent")
    # 에이전트 → tool_call 유무에 따라 분기
    graph.add_conditional_edges("agent", should_use_tools, ["tools", END])
    # 도구 실행 후 다시 에이전트로 (결과를 보고 추가 판단)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ============================================================
# 7단계: 데모 실행
# ============================================================
def main():
    app = build_graph()

    print("=" * 60)
    print("예제 6: SafeToolNode 데모")
    print("=" * 60)

    # 시나리오: 날씨 조회(성공) + 데이터베이스 검색(실패)을 동시에 요청
    # SafeToolNode 덕분에 DB 에러가 발생해도 에이전트가 계속 동작한다
    result = app.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "서울의 날씨를 알려주고, "
                    "데이터베이스에서 '최근 매출 데이터'를 검색해주세요."
                )
            )
        ]
    })

    print(f"\n--- 최종 응답 ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
