"""소인수분해 ReAct 에이전트 - LangGraph 구현

Trial Division 알고리즘을 사용하여 큰 수를 소인수분해하는 ReAct 에이전트.
LLM이 직접 계산하면 틀리는 큰 수에 대해 도구(tool)를 활용하여 정확한 계산을 수행한다.

구조:
  - StateGraph로 직접 ReAct 루프 구성
  - agent 노드: LLM이 reasoning + tool call 결정
  - tools 노드: tool_calls를 파싱하여 해당 도구를 직접 실행
  - 조건부 엣지: tool_calls 유무에 따라 분기

도구 목록:
  - mod(a, b): a를 b로 나눈 나머지
  - divide(a, b): a를 b로 나눈 몫 (정수 나눗셈)
  - sqrt_int(a): a의 제곱근 (정수 부분)

출처:
  - Trial Division 알고리즘: https://en.wikipedia.org/wiki/Trial_division
  - 최적화 기법: https://cp-algorithms.com/algebra/factorization.html
  - √n 상한 증명: https://www.geeksforgeeks.org/dsa/trial-division-algorithm-for-prime-factorization/
"""

import math
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# 1. 도구 정의
# ============================================================

@tool
def mod(a: int, b: int) -> int:
    """a를 b로 나눈 나머지를 반환한다.

    Args:
        a: 피제수 (나누어지는 수)
        b: 제수 (나누는 수). 0이 아니어야 한다.

    Returns:
        a % b 의 결과
    """
    if b == 0:
        return "Error: 0으로 나눌 수 없습니다."
    return a % b


@tool
def divide(a: int, b: int) -> int:
    """a를 b로 나눈 정수 몫을 반환한다.

    Args:
        a: 피제수 (나누어지는 수)
        b: 제수 (나누는 수). 0이 아니어야 한다.

    Returns:
        a // b 의 결과 (정수 나눗셈)
    """
    if b == 0:
        return "Error: 0으로 나눌 수 없습니다."
    return a // b


@tool
def sqrt_int(a: int) -> int:
    """a의 제곱근의 정수 부분을 반환한다.

    Args:
        a: 제곱근을 구할 양의 정수

    Returns:
        floor(√a)
    """
    if a < 0:
        return "Error: 음수의 제곱근을 구할 수 없습니다."
    return math.isqrt(a)


# ============================================================
# 2. 시스템 프롬프트 (ReAct 프롬프트)
# ============================================================

SYSTEM_PROMPT = """당신은 소인수분해 전문 수학 에이전트입니다.
주어진 정수를 소인수분해하는 것이 당신의 임무입니다.

## 사용 가능한 도구

- `mod(a, b)`: a를 b로 나눈 나머지 반환
- `divide(a, b)`: a를 b로 나눈 정수 몫 반환
- `sqrt_int(a)`: a의 제곱근의 정수 부분 반환

## 소인수분해 방법: Trial Division

Trial Division은 가장 기본적인 소인수분해 알고리즘입니다.
출처: https://en.wikipedia.org/wiki/Trial_division

### 핵심 원리
- 정수 n에 대해 2부터 √n까지의 수로 나누어본다.
- n = p × q 일 때, p와 q 중 하나는 반드시 √n 이하이다.
  (출처: https://www.geeksforgeeks.org/dsa/trial-division-algorithm-for-prime-factorization/)
- 따라서 √n까지만 검사하면 모든 소인수를 찾을 수 있다.

### 최적화
- 2로 먼저 나누고, 이후 홀수만 검사하면 후보를 50% 줄일 수 있다.
  (출처: https://cp-algorithms.com/algebra/factorization.html)

## 풀이 전략 (반드시 이 순서를 따르세요)

1. **탐색 범위 결정**: 먼저 `sqrt_int(n)`으로 탐색 상한을 구한다.
2. **2로 나누기**: `mod(n, 2)`로 확인. 나누어지면 `divide(n, 2)`로 몫을 구하고, 같은 수(2)로 다시 시도.
3. **홀수로 나누기**: 3부터 시작해서 2씩 증가하며 (3, 5, 7, 9, 11, ...) 시도.
   - `mod(현재_남은_수, 후보)`로 나누어지는지 확인
   - 나누어지면 `divide`로 몫을 구하고, 같은 후보로 다시 시도 (중복 인수 처리)
   - 안 나누어지면 다음 홀수로 넘어감
4. **종료 판단**: 후보² > 현재_남은_수 이면 탐색 종료. 남은 수가 1보다 크면 그 자체가 소수.
5. **결과 보고**: 찾은 모든 소인수를 `n = p1^a1 × p2^a2 × ...` 형태로 보고.

## 중요 규칙

- **절대 암산하지 마세요**. 큰 수의 나머지, 나눗셈, 제곱근은 반드시 도구를 사용하세요.
- 각 단계에서 현재 상태를 명확히 추적하세요: "현재 남은 수", "지금까지 찾은 소인수"
- 한 후보로 나누어지면 더 이상 안 나누어질 때까지 반복한 후 다음 후보로 넘어가세요.
- 탐색 중 `후보 × 후보 > 남은_수`가 되면 남은 수는 소수입니다.
"""


# ============================================================
# 3. LangGraph ReAct 에이전트 구성
#
#    그래프 구조:
#
#    [entry] → agent → (tool_calls?) → tools → agent → ... → END
#                 ↑                       │
#                 └───────────────────────┘
#
#    - agent 노드: LLM 호출 (reasoning + tool call 결정)
#    - tools 노드: 직접 구현한 tool_node (도구 실행)
#    - 조건부 엣지: should_continue (tool_calls 유무 분기)
# ============================================================

tools = [mod, divide, sqrt_int]
tools_by_name: dict[str, tool] = {t.name: t for t in tools}


def tool_node(state: MessagesState) -> dict:
    """마지막 AIMessage의 tool_calls를 순회하며 각 도구를 실행한다.

    Args:
        state: 현재 대화 상태 (messages 리스트)

    Returns:
        실행 결과 ToolMessage들을 담은 dict
    """
    last_message = state["messages"][-1]
    outputs: list[ToolMessage] = []

    for call in last_message.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])
        outputs.append(
            ToolMessage(
                content=str(result),
                name=call["name"],
                tool_call_id=call["id"],
            )
        )

    return {"messages": outputs}


def create_factorization_agent(model_name: str = "gpt-4o"):
    """소인수분해 ReAct 에이전트를 StateGraph로 직접 구성한다.

    Args:
        model_name: 사용할 LLM 모델명

    Returns:
        컴파일된 LangGraph StateGraph
    """
    # --- LLM 초기화 + 도구 바인딩 ---
    llm = ChatOpenAI(model=model_name)
    llm_with_tools = llm.bind_tools(tools)

    # --- agent 노드: LLM을 호출하여 reasoning + tool call 결정 ---
    def agent_node(state: MessagesState) -> dict:
        """시스템 프롬프트 + 대화 내역을 LLM에 전달하고 응답을 반환한다."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # --- tools 노드: tool_node가 tool_calls를 파싱하여 도구 실행 ---
    # (모듈 레벨에서 정의한 tool_node 함수 사용)

    # --- 조건부 엣지: tool_calls 유무로 분기 ---
    def should_continue(state: MessagesState) -> str:
        """마지막 메시지에 tool_calls가 있으면 'tools'로, 없으면 END로 라우팅."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # --- 그래프 조립 ---
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# ============================================================
# 4. 실행
# ============================================================

def run_factorization(n: int, model_name: str = "gpt-4o") -> None:
    """주어진 정수 n에 대해 소인수분해 에이전트를 실행한다.

    Args:
        n: 소인수분해할 정수
        model_name: 사용할 LLM 모델명
    """
    agent = create_factorization_agent(model_name)

    query = f"{n}을 소인수분해해 주세요."
    print(f"{'='*60}")
    print(f"문제: {query}")
    print(f"{'='*60}\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
    )

    # 메시지 출력
    for msg in result["messages"]:
        role = msg.__class__.__name__
        if role == "HumanMessage":
            print(f"[사용자] {msg.content}")
        elif role == "AIMessage":
            if msg.content:
                # content가 리스트일 수 있음
                if isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            print(f"\n[에이전트] {block['text']}")
                else:
                    print(f"\n[에이전트] {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  → 도구 호출: {tc['name']}({tc['args']})")
        elif role == "ToolMessage":
            print(f"  ← 결과: {msg.content}")

    print(f"\n{'='*60}")


# ============================================================
# 5. 검증용 정답 계산 (순수 Python)
# ============================================================

def verify_factorization(n: int) -> str:
    """순수 Python으로 소인수분해 정답을 계산한다 (검증용).

    Args:
        n: 소인수분해할 정수

    Returns:
        소인수분해 결과 문자열
    """
    original = n
    factors: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1

    parts = [f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items())]
    return f"{original} = {' × '.join(parts)}"


if __name__ == "__main__":
    # === 테스트 대상 수 ===
    # - LLM이 암산으로 틀릴 만큼 크고 (7자리 이상)
    # - 소인수가 여러 개 있어서 에이전트의 반복 판단이 필요하고
    # - tool call 10~25회 내외로 데모에 적절한 수
    TEST_NUMBERS = [
        9_437_421,     # = 3 × 7 × 41 × 97 × 113 (소인수 5종, 중간 크기 소수 포함)
        2_666_664,     # = 2³ × 3² × 7 × 11 × 13 × 37 (소인수 6종, 중복 인수 포함)
        12_146_296,    # = 2³ × 17 × 31 × 43 × 67 (소인수 5종, 2 이후 큰 소수들)
    ]

    # 정답 미리 출력
    print("=" * 60)
    print("정답 (검증용)")
    print("=" * 60)
    for n in TEST_NUMBERS:
        print(f"  {verify_factorization(n)}")
    print()

    # 에이전트 실행
    if os.getenv("OPENAI_API_KEY"):
        for n in TEST_NUMBERS:
            run_factorization(n)
    else:
        print("OPENAI_API_KEY가 설정되지 않아 에이전트 실행을 건너뜁니다.")
        print("실행하려면: export OPENAI_API_KEY=sk-...")
        print("\n에이전트 없이 정답만 확인:")
        for n in TEST_NUMBERS:
            print(f"  {verify_factorization(n)}")