import json

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from state import State


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
