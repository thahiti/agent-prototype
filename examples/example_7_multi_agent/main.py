"""
예제 7: 멀티 에이전트 슈퍼바이저 (Multi-Agent Supervisor)

핵심 아이디어:
- 슈퍼바이저 에이전트가 사용자 요청을 분석·계획하고,
  적절한 워커 에이전트(웹 검색, CLI)에게 작업을 위임한다.
- 각 워커는 독립적인 ReAct 루프 서브그래프로 구성된다.
- 슈퍼바이저는 JSON 출력으로 다음 워커를 결정하고,
  모든 작업 완료 시 최종 답변을 생성한다.

실행 방법:
    uv run examples/example_7_multi_agent/main.py

실행 흐름:
    [START] → [supervisor] → (라우터) → [web_search_agent] → [supervisor] → ...
                                       → [cli_agent]        → [supervisor]
                                       → END (FINISH)
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from state import State
from supervisor import supervisor_node, supervisor_router
from web_search_agent import web_search_wrapper
from cli_agent import cli_wrapper


def build_graph():
    """슈퍼바이저 메인 그래프를 빌드한다."""
    graph = StateGraph(State)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("web_search_agent", web_search_wrapper)
    graph.add_node("cli_agent", cli_wrapper)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        ["web_search_agent", "cli_agent", END],
    )
    graph.add_edge("web_search_agent", "supervisor")
    graph.add_edge("cli_agent", "supervisor")

    return graph.compile()


def main():
    app = build_graph()

    print("=" * 60)
    print("예제 7: 멀티 에이전트 슈퍼바이저 데모")
    print("=" * 60)

    # 시나리오 A: 웹 검색만
    print("\n" + "▶" * 30)
    print("시나리오 A: 웹 검색만")
    print("▶" * 30)

    result_a = app.invoke({
        "messages": [HumanMessage(content="Python 3.12의 새 기능을 검색해주세요")],
        "next_agent": "",
        "plan": "",
        "completed_agents": [],
    })

    print(f"\n--- 시나리오 A 최종 응답 ---")
    print(result_a["messages"][-1].content)

    # 시나리오 B: CLI만
    print("\n" + "▶" * 30)
    print("시나리오 B: CLI만")
    print("▶" * 30)

    result_b = app.invoke({
        "messages": [HumanMessage(content="현재 시스템의 Python 버전과 오늘 날짜를 확인해주세요")],
        "next_agent": "",
        "plan": "",
        "completed_agents": [],
    })

    print(f"\n--- 시나리오 B 최종 응답 ---")
    print(result_b["messages"][-1].content)

    # 시나리오 C: 복합 요청 (웹 검색 + CLI)
    print("\n" + "▶" * 30)
    print("시나리오 C: 복합 요청 (웹 검색 + CLI)")
    print("▶" * 30)

    result_c = app.invoke({
        "messages": [HumanMessage(content="Python 3.12 새 기능을 검색하고, 제 시스템의 Python 버전도 확인해주세요")],
        "next_agent": "",
        "plan": "",
        "completed_agents": [],
    })

    print(f"\n--- 시나리오 C 최종 응답 ---")
    print(result_c["messages"][-1].content)


if __name__ == "__main__":
    main()
