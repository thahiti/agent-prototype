from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from state import State, WorkerState


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


web_search_subgraph = build_web_search_agent()


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
