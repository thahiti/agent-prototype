"""
예제 2: Human-in-the-loop (HITL) 공통 구조

핵심 아이디어:
- 에이전트가 위험한 작업(예: 이메일 발송, 결제)을 수행하기 전에
  사람의 승인을 받도록 강제하는 구조.
- LangGraph의 interrupt() 함수로 그래프 실행을 일시 중지하고,
  Command(resume=...)로 재개한다.
- Checkpointer(MemorySaver)가 중단 시점의 상태를 보존한다.

실행 방법:
    uv run examples/example_2_human_in_the_loop/main.py

실행 흐름:
    [START] → [agent_node] → (라우터) → [human_review_node] → [execute_node] → [END]
                                ↘ (승인 불필요) → [END]
"""

from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# 1단계: HITL 전용 State 정의
# ============================================================
class State(TypedDict):
    """Human-in-the-loop에 필요한 필드를 포함한 상태.
    - messages: 대화 이력
    - requires_human_approval: 사람의 승인이 필요한지 여부
    - pending_action: 승인 대기 중인 작업 설명
    """
    messages: Annotated[list[BaseMessage], add_messages]
    requires_human_approval: bool
    pending_action: str


# ============================================================
# 2단계: 에이전트 노드 - 작업을 판단하고 승인 필요 여부를 결정
# ============================================================
def agent_node(state: State) -> dict:
    """에이전트가 사용자 요청을 분석하여
    승인이 필요한 작업인지 판단하는 노드.

    실제로는 LLM이 판단하지만, 여기서는 핵심 흐름을 보이기 위해
    '발송', '삭제', '결제' 키워드로 단순 판별한다.
    """
    last_message = state["messages"][-1].content

    # 위험한 키워드가 포함되면 승인 필요
    dangerous_keywords = ["발송", "삭제", "결제", "전송"]
    needs_approval = any(kw in last_message for kw in dangerous_keywords)

    if needs_approval:
        return {
            "messages": [AIMessage(content=f"작업 확인: '{last_message}' — 승인이 필요합니다.")],
            "requires_human_approval": True,
            "pending_action": last_message,
        }
    else:
        return {
            "messages": [AIMessage(content=f"작업 완료: '{last_message}' — 즉시 처리했습니다.")],
            "requires_human_approval": False,
            "pending_action": "",
        }


# ============================================================
# 3단계: 라우터 - 승인 필요 여부에 따라 분기
# ============================================================
def approval_router(state: State) -> str:
    """conditional_edge에서 사용하는 라우팅 함수.
    requires_human_approval 값에 따라 다음 노드를 결정한다.
    """
    if state.get("requires_human_approval", False):
        return "human_review"  # 승인 노드로 이동
    return END  # 바로 종료


# ============================================================
# 4단계: 사람 리뷰 노드 - interrupt()로 실행 일시 중지
# ============================================================
def human_review_node(state: State) -> Command:
    """이 노드에서 interrupt()를 호출하여 그래프 실행을 멈춘다.
    외부(사용자)에서 Command(resume=True/False)로 재개한다.

    interrupt()의 동작 원리:
    1. 이 함수가 호출되면 그래프 실행이 즉시 멈춘다.
    2. Checkpointer가 현재 상태를 저장한다.
    3. 사용자가 승인/거부를 결정한 후 Command(resume=값)으로 재개한다.
    4. interrupt()는 resume에 전달된 값을 반환한다.
    """
    print(f"\n⏸️  승인 대기 중: '{state['pending_action']}'")
    print("   (외부에서 Command(resume=True)로 승인, resume=False로 거부)")

    # interrupt()가 호출되면 여기서 실행이 멈춘다.
    # 재개 시 human_decision에는 resume 값(True/False)이 들어온다.
    human_decision = interrupt(
        {
            "question": f"다음 작업을 승인하시겠습니까? → '{state['pending_action']}'",
            "options": ["승인(True)", "거부(False)"],
        }
    )

    if human_decision:
        # 승인됨 → execute_node로 이동
        return Command(
            update={
                "messages": [AIMessage(content="✅ 사람이 승인했습니다. 작업을 실행합니다.")],
            },
            goto="execute",
        )
    else:
        # 거부됨 → 종료
        return Command(
            update={
                "messages": [AIMessage(content="❌ 사람이 거부했습니다. 작업을 취소합니다.")],
                "requires_human_approval": False,
                "pending_action": "",
            },
            goto=END,
        )


# ============================================================
# 5단계: 실행 노드 - 승인된 작업을 실제로 수행
# ============================================================
def execute_node(state: State) -> dict:
    """승인된 작업을 실제로 실행하는 노드.
    실제로는 이메일 발송, API 호출 등이 이루어진다.
    """
    action = state.get("pending_action", "알 수 없는 작업")
    print(f"\n🚀 작업 실행 중: '{action}'")

    return {
        "messages": [AIMessage(content=f"작업 '{action}'이(가) 성공적으로 실행되었습니다.")],
        "requires_human_approval": False,
        "pending_action": "",
    }


# ============================================================
# 6단계: 그래프 조립 및 실행
# ============================================================
def build_graph():
    """HITL 그래프를 조립한다.
    핵심: MemorySaver를 checkpointer로 사용하여
    interrupt() 시점의 상태를 보존한다.
    """
    graph = StateGraph(State)

    # 노드 등록
    graph.add_node("agent", agent_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("execute", execute_node)

    # 엣지 연결
    graph.add_edge(START, "agent")
    # 에이전트 → 라우터(조건부 분기)
    graph.add_conditional_edges("agent", approval_router, ["human_review", END])
    graph.add_edge("execute", END)

    # MemorySaver: 인메모리 체크포인터 (interrupt 상태 저장용)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def main():
    app = build_graph()

    print("=" * 50)
    print("예제 2: Human-in-the-loop 데모")
    print("=" * 50)

    # 대화 세션을 식별하기 위한 config (thread_id 필수)
    config = {"configurable": {"thread_id": "demo-session-1"}}

    # --- 시나리오 A: 승인이 필요 없는 작업 ---
    print("\n--- 시나리오 A: 일반 작업 (승인 불필요) ---")
    result_a = app.invoke(
        {"messages": [HumanMessage(content="오늘 날씨 알려줘")]},
        config=config,
    )
    print(f"결과: {result_a['messages'][-1].content}")

    # --- 시나리오 B: 승인이 필요한 작업 ---
    print("\n--- 시나리오 B: 위험한 작업 (승인 필요) ---")
    config_b = {"configurable": {"thread_id": "demo-session-2"}}

    # 1) 첫 실행: interrupt()에서 멈춘다
    result_b = app.invoke(
        {"messages": [HumanMessage(content="고객에게 이메일 발송해줘")]},
        config=config_b,
    )
    print(f"중단 시점 메시지: {result_b['messages'][-1].content}")

    # 2) 사람이 승인하여 재개
    print("\n--- 사람이 '승인'을 클릭 ---")
    result_b2 = app.invoke(
        Command(resume=True),  # True = 승인
        config=config_b,
    )
    print(f"최종 결과: {result_b2['messages'][-1].content}")


if __name__ == "__main__":
    main()
