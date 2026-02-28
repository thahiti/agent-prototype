from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


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


class WorkerState(TypedDict):
    """워커 서브그래프의 내부 상태."""
    messages: Annotated[list[BaseMessage], add_messages]
