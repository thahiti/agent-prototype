"""
예제 4: 컨텍스트 윈도우 관리기 (State Trimming)

핵심 아이디어:
- 대화가 길어지면 토큰이 누적되어 비용 증가 + 컨텍스트 초과 에러가 발생한다.
- 그래프 진입점(START 직후)에 트리밍 노드를 배치하여,
  LLM 호출 전에 메시지를 자동으로 정리한다.
- 두 가지 전략을 제공:
  1) 단순 트리밍: 최근 N개 메시지 + SystemMessage만 유지
  2) 요약 트리밍: 오래된 메시지를 LLM으로 요약하여 대체

실행 방법:
    uv run examples/example_4_context_trimmer/main.py

실행 흐름:
    [START] → [trim_messages_node] → [chatbot] → [END]
"""

import tiktoken
from typing import TypedDict, Annotated

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1단계: State 정의
# ============================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 2단계: 토큰 수 계산 유틸리티
# ============================================================
def count_tokens(messages: list[BaseMessage], model: str = "gpt-4o-mini") -> int:
    """메시지 리스트의 총 토큰 수를 계산한다.

    tiktoken 라이브러리를 사용하여 실제 OpenAI 모델의
    토크나이저와 동일한 방식으로 토큰을 센다.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        # 메시지 오버헤드 (~4 토큰: role, content 구분자 등)
        total += 4
        total += len(encoding.encode(msg.content))
    return total


# ============================================================
# 3단계: 트리밍 노드 구현
# ============================================================
def trim_messages_node(state: State) -> dict:
    """메시지 리스트가 토큰 임계치를 초과하면 트리밍한다.

    전략:
    1. SystemMessage는 항상 보존 (에이전트의 역할 정의)
    2. 최근 keep_last_n개의 메시지는 항상 유지
    3. 임계치 초과 시 중간 메시지를 삭제하고 요약으로 대체

    이 노드는 그래프의 START 직후에 배치되어
    모든 LLM 호출 전에 컨텍스트를 정리한다.
    """
    messages = state["messages"]
    max_tokens = 500       # 데모용 낮은 임계치 (실제로는 4000~8000)
    keep_last_n = 4        # 최근 4개 메시지는 항상 유지

    current_tokens = count_tokens(messages)
    print(f"[TRIMMER] 현재 토큰 수: {current_tokens} / 임계치: {max_tokens}")

    # 임계치 이하이면 트리밍하지 않음
    if current_tokens <= max_tokens:
        print(f"[TRIMMER] 임계치 이내 → 트리밍 불필요")
        return {"messages": messages}

    print(f"[TRIMMER] 임계치 초과! 트리밍 시작...")

    # --- SystemMessage 분리 ---
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

    # --- 최근 N개 메시지 보존 ---
    if len(non_system_messages) <= keep_last_n:
        # 메시지 수가 적으면 트리밍할 것이 없음
        print(f"[TRIMMER] 메시지 수({len(non_system_messages)})가 "
              f"keep_last_n({keep_last_n}) 이하 → 트리밍 생략")
        return {"messages": messages}

    recent_messages = non_system_messages[-keep_last_n:]
    old_messages = non_system_messages[:-keep_last_n]

    # --- 오래된 메시지를 요약으로 대체 ---
    summary = summarize_messages(old_messages)

    # 최종 메시지 리스트: System + 요약 + 최근 메시지
    trimmed = (
        system_messages
        + [SystemMessage(content=f"[이전 대화 요약] {summary}")]
        + recent_messages
    )

    new_tokens = count_tokens(trimmed)
    print(f"[TRIMMER] 트리밍 완료: {current_tokens} → {new_tokens} 토큰")
    print(f"[TRIMMER] 삭제된 메시지: {len(old_messages)}개 → 요약 1개로 대체")

    return {"messages": trimmed}


def summarize_messages(messages: list[BaseMessage]) -> str:
    """오래된 메시지를 LLM으로 요약한다.

    실제 프로덕션에서는 저렴한 모델(gpt-4o-mini 등)을 사용하여
    비용을 절감한다.
    """
    if not messages:
        return "이전 대화 없음"

    # 대화 내용을 텍스트로 변환
    conversation_text = "\n".join(
        f"{'사용자' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in messages
    )

    # 요약용 LLM 호출 (저렴한 모델 사용)
    summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    summary_response = summarizer.invoke([
        HumanMessage(
            content=f"다음 대화를 2~3문장으로 요약하세요:\n\n{conversation_text}"
        )
    ])

    return summary_response.content


# ============================================================
# 4단계: 챗봇 노드
# ============================================================
def chatbot(state: State) -> dict:
    """트리밍된 메시지를 받아 LLM을 호출하는 노드."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ============================================================
# 5단계: 그래프 조립 — 트리머를 START 직후에 배치
# ============================================================
def build_graph():
    """START → trim_messages → chatbot → END
    트리머가 항상 LLM 호출 전에 실행되어 컨텍스트를 관리한다.
    """
    graph = StateGraph(State)
    graph.add_node("trimmer", trim_messages_node)
    graph.add_node("chatbot", chatbot)

    graph.add_edge(START, "trimmer")
    graph.add_edge("trimmer", "chatbot")
    graph.add_edge("chatbot", END)

    return graph.compile()


# ============================================================
# 6단계: 데모 실행 — 긴 대화를 시뮬레이션
# ============================================================
def main():
    app = build_graph()

    print("=" * 60)
    print("예제 4: 컨텍스트 윈도우 관리기 데모")
    print("=" * 60)

    # 긴 대화 시뮬레이션: 메시지를 많이 쌓아서 트리밍을 유도
    long_conversation = [
        SystemMessage(content="당신은 친절한 AI 비서입니다."),
        HumanMessage(content="안녕하세요! Python에 대해 알려주세요."),
        AIMessage(content="안녕하세요! Python은 간결하고 읽기 쉬운 프로그래밍 언어입니다."),
        HumanMessage(content="Python의 주요 특징은 뭔가요?"),
        AIMessage(content="동적 타이핑, 풍부한 라이브러리, 가독성 높은 문법이 특징입니다."),
        HumanMessage(content="데이터 분석에 많이 쓰이나요?"),
        AIMessage(content="네, pandas, numpy, matplotlib 등으로 데이터 분석에 널리 사용됩니다."),
        HumanMessage(content="웹 개발에도 사용되나요?"),
        AIMessage(content="Django와 Flask 같은 프레임워크로 웹 개발에도 많이 활용됩니다."),
        HumanMessage(content="머신러닝은요?"),
        AIMessage(content="TensorFlow, PyTorch, scikit-learn 등으로 머신러닝의 표준 언어입니다."),
        # 마지막 질문 — 이 시점에서 트리밍이 발동
        HumanMessage(content="그렇다면 Python으로 LLM 에이전트를 만들 수 있나요?"),
    ]

    print(f"\n입력 메시지 수: {len(long_conversation)}")
    print(f"입력 토큰 수: {count_tokens(long_conversation)}")

    # 그래프 실행 → 트리머가 먼저 실행되어 메시지를 정리한 후 챗봇 호출
    result = app.invoke({"messages": long_conversation})

    print(f"\n--- 최종 응답 ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
