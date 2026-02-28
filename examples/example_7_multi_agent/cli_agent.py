import subprocess

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from state import State, WorkerState


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


cli_subgraph = build_cli_agent()


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
