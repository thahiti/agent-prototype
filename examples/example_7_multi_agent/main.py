"""
예제 7: 멀티 에이전트 슈퍼바이저 (Multi-Agent Supervisor)

핵심 아이디어:
- 슈퍼바이저 에이전트가 사용자 요청을 분석·계획하고,
  적절한 워커 에이전트(웹 검색, CLI)에게 작업을 위임한다.
- 각 워커는 독립적인 ReAct 루프 서브그래프로 구성된다.
- 슈퍼바이저는 JSON 출력으로 다음 워커를 결정하고,
  모든 작업 완료 시 최종 답변을 생성한다.

실행 흐름:
    [START] → [supervisor] → (라우터) → [web_search_agent] → [supervisor] → ...
                                       → [cli_agent]        → [supervisor]
                                       → END (FINISH)
"""

import json
import subprocess
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# 1단계: State 정의
# ============================================================
class State(TypedDict):
    """멀티 에이전트 시스템의 공유 상태.

    messages: 전체 대화 이력 (add_messages 리듀서로 자동 병합)
    next_agent: 슈퍼바이저가 결정한 다음 워커 ("web_search" | "cli" | "FINISH")
    plan: 슈퍼바이저의 실행 계획
    completed_agents: 작업을 완료한 워커 목록
    """
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    plan: str
    completed_agents: list[str]


# ============================================================
# 2단계: 도구 정의
# ============================================================
@tool
def web_search(query: str) -> str:
    """인터넷에서 정보를 검색한다. 검색어를 입력하면 관련 결과를 반환한다."""
    from langchain_tavily import TavilySearch

    tavily = TavilySearch(max_results=3)
    results = tavily.invoke(query)

    if isinstance(results, list):
        formatted = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            formatted.append(f"제목: {title}\n내용: {content}\nURL: {url}")
        return "\n\n---\n\n".join(formatted)
    return str(results)


ALLOWED_COMMANDS = {"python", "python3", "node", "echo", "cat", "ls", "pwd", "whoami", "date", "uname"}
BLOCKED_PATTERNS = ["rm ", "sudo ", "chmod ", "chown ", "mkfs ", "dd if=", "rm\t", "&&", "||", "|", ";", ">", "<"]
CLI_TIMEOUT = 10


@tool
def run_cli_command(command: str) -> str:
    """로컬 셸에서 명령어를 실행한다. 안전한 명령어만 허용된다.

    허용 명령어: python, python3, node, echo, cat, ls, pwd, whoami, date, uname
    """
    parts = command.strip().split()
    if not parts:
        return "[CLI 에러] 빈 명령어입니다."

    base_command = parts[0]
    if base_command not in ALLOWED_COMMANDS:
        return f"[CLI 에러] 허용되지 않은 명령어: '{base_command}'. 허용 목록: {', '.join(sorted(ALLOWED_COMMANDS))}"

    for pattern in BLOCKED_PATTERNS:
        if pattern in command:
            return f"[CLI 에러] 차단된 패턴 감지: '{pattern}'"

    try:
        result = subprocess.run(
            command.strip().split(),
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode != 0:
            return f"[CLI 에러] 종료 코드 {result.returncode}\nstdout: {output}\nstderr: {error}"
        return output if output else "(출력 없음)"

    except subprocess.TimeoutExpired:
        return f"[CLI 에러] 명령어 실행 시간 초과 ({CLI_TIMEOUT}초)"
    except Exception as e:
        return f"[CLI 에러] {type(e).__name__}: {str(e)}"


# ============================================================
# 3단계: 워커 서브그래프 (ReAct 루프)
# ============================================================
class WorkerState(TypedDict):
    """워커 서브그래프의 내부 상태."""
    messages: Annotated[list[BaseMessage], add_messages]


def build_web_search_agent():
    """웹 검색 에이전트 서브그래프를 빌드한다."""
    search_tools = [web_search]

    def web_search_agent_node(state: WorkerState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm_with_tools = llm.bind_tools(search_tools)

        system_msg = SystemMessage(content=(
            "당신은 웹 검색 전문 에이전트입니다.\n"
            "사용자의 질문에 답하기 위해 web_search 도구를 사용하여 인터넷에서 정보를 검색하세요.\n"
            "검색 결과를 바탕으로 정확하고 유용한 답변을 한국어로 작성하세요."
        ))

        messages = [system_msg] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def web_search_should_continue(state: WorkerState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(WorkerState)
    tool_node = ToolNode(search_tools)

    graph.add_node("agent", web_search_agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", web_search_should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()


def build_cli_agent():
    """CLI 에이전트 서브그래프를 빌드한다."""
    cli_tools = [run_cli_command]

    def cli_agent_node(state: WorkerState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm_with_tools = llm.bind_tools(cli_tools)

        system_msg = SystemMessage(content=(
            "당신은 CLI(명령줄) 전문 에이전트입니다.\n"
            "사용자의 요청을 처리하기 위해 run_cli_command 도구를 사용하여 셸 명령어를 실행하세요.\n"
            "허용 명령어: python, python3, node, echo, cat, ls, pwd, whoami, date, uname\n"
            "결과를 한국어로 정리하여 보고하세요."
        ))

        messages = [system_msg] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def cli_should_continue(state: WorkerState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(WorkerState)
    tool_node = ToolNode(cli_tools)

    graph.add_node("agent", cli_agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", cli_should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()


# ============================================================
# 4단계: 슈퍼바이저 구현
# ============================================================
SUPERVISOR_SYSTEM_PROMPT = """당신은 멀티 에이전트 시스템의 슈퍼바이저입니다.

사용자의 요청을 분석하고, 적절한 워커 에이전트에게 작업을 위임하세요.

## 사용 가능한 워커:
- **web_search**: 인터넷 검색으로 정보를 수집합니다. 최신 정보, 뉴스, 기술 문서 등을 찾을 때 사용합니다.
- **cli**: 로컬 셸 명령어를 실행합니다. 시스템 정보 확인, 파일 조회, Python/Node 실행 등에 사용합니다.

## 현재 상태:
- 실행 계획: {plan}
- 완료된 워커: {completed_agents}

## 응답 형식:
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

```json
{{"next": "web_search" 또는 "cli" 또는 "FINISH", "reason": "이유 설명", "plan": "전체 실행 계획"}}
```

## 규칙:
1. 모든 필요한 작업이 완료되면 "FINISH"를 선택하세요.
2. 각 워커는 한 번에 하나의 작업만 처리합니다.
3. 계획을 세우고 순서대로 워커를 호출하세요.
4. FINISH 선택 시, 워커들의 결과를 종합하여 reason에 최종 요약을 포함하세요."""


MAX_ITERATIONS = 5


def extract_json_from_text(text: str) -> dict:
    """텍스트에서 JSON을 추출한다. ```json 블록 또는 직접 파싱을 시도한다."""
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + len("```")
        end = text.index("```", start)
        text = text[start:end].strip()
    return json.loads(text)


def supervisor_node(state: State) -> dict:
    """슈퍼바이저: 요청 분석, 계획 수립, 다음 워커 결정."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    plan = state.get("plan", "아직 계획 없음")
    completed = state.get("completed_agents", [])

    system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
        plan=plan,
        completed_agents=", ".join(completed) if completed else "없음",
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    content = response.content

    print(f"\n{'='*50}")
    print(f"[SUPERVISOR] 응답: {content}")

    try:
        parsed = extract_json_from_text(content)
        next_agent = parsed.get("next", "FINISH")
        reason = parsed.get("reason", "")
        new_plan = parsed.get("plan", plan)

        print(f"[SUPERVISOR] 다음 에이전트: {next_agent}")
        print(f"[SUPERVISOR] 이유: {reason}")
        print(f"[SUPERVISOR] 계획: {new_plan}")
        print(f"{'='*50}")

        return {
            "messages": [AIMessage(content=content)],
            "next_agent": next_agent,
            "plan": new_plan,
        }

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[SUPERVISOR] ⚠️ JSON 파싱 실패: {e} → FINISH로 안전 종료")
        print(f"{'='*50}")
        return {
            "messages": [AIMessage(content=content)],
            "next_agent": "FINISH",
            "plan": plan,
        }


def supervisor_router(state: State) -> str:
    """슈퍼바이저의 결정에 따라 다음 노드를 라우팅한다."""
    next_agent = state.get("next_agent", "FINISH")

    completed = state.get("completed_agents", [])
    if len(completed) >= MAX_ITERATIONS:
        print(f"[ROUTER] ⚠️ 최대 반복 횟수({MAX_ITERATIONS}) 도달 → END")
        return END

    if next_agent == "web_search":
        print(f"[ROUTER] → web_search_agent")
        return "web_search_agent"
    elif next_agent == "cli":
        print(f"[ROUTER] → cli_agent")
        return "cli_agent"
    else:
        print(f"[ROUTER] → END (FINISH)")
        return END


# ============================================================
# 5단계: 워커 래퍼 함수
# ============================================================
web_search_subgraph = build_web_search_agent()
cli_subgraph = build_cli_agent()


def web_search_wrapper(state: State) -> dict:
    """웹 검색 서브그래프를 실행하고 completed_agents를 업데이트한다."""
    print(f"\n{'─'*40}")
    print(f"[WEB_SEARCH] 웹 검색 에이전트 시작")
    print(f"{'─'*40}")

    result = web_search_subgraph.invoke({"messages": state["messages"]})

    last_message = result["messages"][-1]
    completed = list(state.get("completed_agents", []))
    completed.append("web_search")

    print(f"[WEB_SEARCH] 완료. 결과: {last_message.content[:100]}...")

    return {
        "messages": [AIMessage(content=f"[웹 검색 결과]\n{last_message.content}")],
        "completed_agents": completed,
    }


def cli_wrapper(state: State) -> dict:
    """CLI 서브그래프를 실행하고 completed_agents를 업데이트한다."""
    print(f"\n{'─'*40}")
    print(f"[CLI] CLI 에이전트 시작")
    print(f"{'─'*40}")

    result = cli_subgraph.invoke({"messages": state["messages"]})

    last_message = result["messages"][-1]
    completed = list(state.get("completed_agents", []))
    completed.append("cli")

    print(f"[CLI] 완료. 결과: {last_message.content[:100]}...")

    return {
        "messages": [AIMessage(content=f"[CLI 실행 결과]\n{last_message.content}")],
        "completed_agents": completed,
    }


# ============================================================
# 6단계: 메인 그래프 조립
# ============================================================
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


# ============================================================
# 7단계: 데모 실행
# ============================================================
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
