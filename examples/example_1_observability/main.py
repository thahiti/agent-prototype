"""
예제 1: Observability (관측성) 래퍼 노드

핵심 아이디어:
- 파이썬 데코레이터로 노드의 실행 시간, 상태 변경점(Delta)을 자동 기록한다.
- 비즈니스 로직 코드를 전혀 수정하지 않고도 모니터링을 추가할 수 있다.
- 실제 프로덕션에서는 수집된 데이터를 외부 시스템(GaiA, Datadog 등)으로 전송한다.

실행 방법:
    uv run examples/example_1_observability/main.py

실행 흐름:
    [START] → [chatbot (with @observe_agent)] → [END]
"""

import time
import json
import functools
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1단계: 공통 State 정의
# ============================================================
class State(TypedDict):
    """그래프 전체에서 공유되는 상태.
    messages 필드에 add_messages 리듀서를 붙여서
    새 메시지가 기존 리스트에 '추가'되도록 한다.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 2단계: @observe_agent 데코레이터 구현
# ============================================================
def observe_agent(agent_name: str):
    """노드 함수를 감싸서 실행 시간과 상태 변경점을 로깅하는 데코레이터.

    사용법:
        @observe_agent(agent_name="chatbot")
        def chatbot(state: State) -> dict:
            ...

    동작 원리:
        1. 노드 실행 전 시간과 상태 스냅샷을 저장한다.
        2. 원본 노드 함수를 실행한다.
        3. 실행 후 걸린 시간과 상태 변경점(Delta)을 계산하여 출력한다.
    """
    def decorator(func):
        @functools.wraps(func)  # 원본 함수의 __name__, __doc__ 등 메타데이터 유지
        def wrapper(state: State):
            # --- 실행 전 ---
            start_time = time.time()
            # 실행 전 메시지 개수를 스냅샷으로 저장
            before_count = len(state.get("messages", []))

            print(f"\n{'='*50}")
            print(f"[OBSERVE] '{agent_name}' 노드 실행 시작")
            print(f"[OBSERVE] 현재 메시지 수: {before_count}")

            # --- 원본 노드 실행 ---
            result = func(state)

            # --- 실행 후 ---
            elapsed = time.time() - start_time
            # 결과에서 새로 추가된 메시지 수 계산
            new_messages = result.get("messages", [])
            after_count = before_count + len(new_messages)

            # Delta(변경점) 정보 구성
            delta = {
                "agent": agent_name,
                "elapsed_seconds": round(elapsed, 3),
                "messages_before": before_count,
                "messages_after": after_count,
                "new_messages_added": len(new_messages),
            }

            print(f"[OBSERVE] '{agent_name}' 노드 실행 완료")
            print(f"[OBSERVE] Delta: {json.dumps(delta, ensure_ascii=False, indent=2)}")
            print(f"{'='*50}\n")

            # 실제 프로덕션에서는 여기서 비동기로 외부 시스템에 전송:
            # asyncio.create_task(send_to_monitoring(delta))

            return result

        return wrapper
    return decorator


# ============================================================
# 3단계: 실제 노드 함수에 데코레이터 적용
# ============================================================
@observe_agent(agent_name="chatbot")
def chatbot(state: State) -> dict:
    """간단한 챗봇 노드. LLM을 호출하여 응답을 생성한다.
    @observe_agent 덕분에 이 함수 자체는 비즈니스 로직만 담당한다.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ============================================================
# 4단계: 그래프 구성 및 실행
# ============================================================
def build_graph():
    """가장 단순한 구조: START → chatbot → END"""
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    return graph.compile()


def main():
    app = build_graph()

    print("=" * 50)
    print("예제 1: Observability 데코레이터 데모")
    print("=" * 50)

    # 테스트 실행
    result = app.invoke({
        "messages": [HumanMessage(content="LangGraph가 뭔가요? 한 문장으로 답해주세요.")]
    })

    # 최종 결과 출력
    print("\n--- 최종 응답 ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
