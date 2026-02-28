# 예제 7: 멀티 에이전트 슈퍼바이저 — 리서치 문서

## 1. 개요

본 예제는 **슈퍼바이저 패턴**을 사용하여 여러 전문 워커 에이전트를 오케스트레이션하는 멀티 에이전트 시스템을 구현한다. 슈퍼바이저가 중앙 컨트롤러 역할을 하며, 사용자 요청을 분석·계획하고 적절한 워커에게 작업을 위임한 뒤, 최종 결과를 종합하여 응답한다.

### 왜 멀티 에이전트인가?

단일 에이전트는 모든 도구를 직접 호출해야 하므로 시스템 프롬프트가 비대해지고, 도구 선택 정확도가 떨어진다. 멀티 에이전트 아키텍처는 각 에이전트에게 제한된 도구와 명확한 역할을 부여하여 이 문제를 해결한다.

| 비교 항목 | 단일 에이전트 | 멀티 에이전트 (슈퍼바이저) |
|-----------|--------------|---------------------------|
| 도구 관리 | 모든 도구를 하나의 LLM에 바인딩 | 워커별로 관련 도구만 바인딩 |
| 프롬프트 복잡도 | 높음 (모든 지침 포함) | 낮음 (역할별 분리) |
| 확장성 | 도구 추가 시 프롬프트 재설계 필요 | 새 워커 에이전트만 추가 |
| 에러 격리 | 하나의 도구 실패가 전체에 영향 | 워커별 독립 실행, 실패 격리 |

---

## 2. 아키텍처

### 2.1 전체 그래프 구조

```
[START]
  │
  ▼
[supervisor] ◄───────────────────────┐
  │                                  │
  ├─ next="web_search" ──► [web_search_agent] ──┤
  │                                              │
  ├─ next="cli" ──────────► [cli_agent] ────────┘
  │
  └─ next="FINISH" ──► [END]
```

슈퍼바이저는 **허브-앤-스포크(hub-and-spoke)** 토폴로지의 중심이다. 모든 워커는 작업 완료 후 반드시 슈퍼바이저로 복귀하며, 슈퍼바이저가 다음 행동(다른 워커 호출 또는 종료)을 결정한다.

### 2.2 워커 서브그래프 내부 구조

각 워커는 독립적인 **ReAct(Reasoning + Acting) 루프**로 구성된다:

```
[START] → [agent_llm] ──► tool_calls 있음? ──► [tool_node] ──► [agent_llm]
               │
               └──► tool_calls 없음? ──► [END]
```

- `agent_llm`: LLM이 도구 호출 여부를 판단하고, 필요 시 `tool_calls`를 생성
- `tool_node`: `ToolNode`가 도구를 실행하고 결과를 `ToolMessage`로 반환
- 도구 호출이 없으면 최종 텍스트 응답으로 간주하고 종료

### 2.3 데이터 흐름

```
사용자 입력
    │
    ▼
State.messages에 HumanMessage 추가
    │
    ▼
supervisor_node: SystemMessage + messages → LLM → JSON 파싱
    │
    ├─ State.next_agent = "web_search" | "cli" | "FINISH"
    ├─ State.plan = "실행 계획 텍스트"
    │
    ▼
supervisor_router: next_agent 값에 따라 분기
    │
    ▼
워커 래퍼 (web_search_wrapper / cli_wrapper):
    ├─ 부모 State.messages → WorkerState.messages로 전달
    ├─ 서브그래프 실행 (ReAct 루프)
    ├─ 결과를 AIMessage로 감싸서 부모 State.messages에 추가
    └─ State.completed_agents에 워커 이름 추가
    │
    ▼
supervisor로 복귀 → 다음 결정 반복
```

---

## 3. 핵심 컴포넌트 상세

### 3.1 State 설계

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 대화 이력
    next_agent: str          # 라우팅 결정
    plan: str                # 실행 계획
    completed_agents: list[str]  # 완료 워커 추적
```

**설계 결정:**
- `messages`에 `add_messages` 리듀서를 사용하여 메시지가 덮어쓰기 되지 않고 누적된다.
- `completed_agents`는 리듀서 없이 직접 리스트를 복사·추가하는 방식이다. 이는 슈퍼바이저가 완료 현황을 파악하여 중복 호출을 방지하는 데 사용된다.
- `next_agent`는 라우터가 참조하는 일시적 상태로, 슈퍼바이저 노드에서만 갱신된다.

**워커 내부 상태:**
```python
class WorkerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

워커는 `messages`만 필요하므로 별도의 간소한 State를 정의했다. 부모 State와 다른 TypedDict를 사용하여 서브그래프의 관심사를 분리한다.

### 3.2 슈퍼바이저 라우팅

슈퍼바이저의 핵심 역할은 **구조화된 JSON 출력으로 다음 행동을 결정**하는 것이다.

**시스템 프롬프트 구조:**
1. 사용 가능한 워커 목록과 각 워커의 역할 설명
2. 현재 실행 상태 (계획, 완료된 워커) — 동적으로 주입
3. 응답 형식 제약 (JSON 스키마)
4. 라우팅 규칙 (완료 시 FINISH, 순차 호출 등)

**JSON 파싱 전략 (example_5 패턴 재사용):**
```python
def extract_json_from_text(text: str) -> dict:
    # 1순위: ```json ... ``` 코드 블록 추출
    # 2순위: ``` ... ``` 일반 코드 블록 추출
    # 3순위: 전체 텍스트를 직접 json.loads 시도
    # 실패 시: FINISH로 안전 폴백
```

이 3단계 파싱 전략은 LLM이 다양한 형식으로 JSON을 출력하더라도 최대한 추출할 수 있도록 설계되었다. 파싱 자체가 실패하면 시스템이 무한 루프에 빠지지 않도록 `FINISH`로 안전하게 종료한다.

**무한 루프 방지:**
```python
MAX_ITERATIONS = 5

def supervisor_router(state: State) -> str:
    if len(state.get("completed_agents", [])) >= MAX_ITERATIONS:
        return END  # 강제 종료
```

### 3.3 웹 검색 도구 (Tavily)

```python
@tool
def web_search(query: str) -> str:
    from langchain_tavily import TavilySearch
    tavily = TavilySearch(max_results=3)
    results = tavily.invoke(query)
    # 결과를 "제목 / 내용 / URL" 형식으로 포맷팅하여 반환
```

**설계 결정:**
- `TavilySearch`를 함수 내부에서 임포트하여 `TAVILY_API_KEY` 미설정 시에도 모듈 로딩이 실패하지 않도록 한다.
- `max_results=3`으로 제한하여 컨텍스트 윈도우 소비를 줄인다.
- 결과를 구조화된 텍스트로 포맷팅하여 LLM이 쉽게 해석할 수 있도록 한다.

### 3.4 CLI 도구 (보안 설계)

CLI 도구는 로컬 시스템에서 셸 명령어를 실행하므로 **보안이 핵심**이다.

**3중 보안 레이어:**

| 레이어 | 메커니즘 | 예시 |
|--------|---------|------|
| 1. 허용 목록 | `ALLOWED_COMMANDS` 집합에 포함된 명령어만 실행 | `python`, `date`, `ls` 등 10개 |
| 2. 차단 패턴 | `BLOCKED_PATTERNS`에 위험 패턴 포함 여부 검사 | `rm`, `sudo`, `chmod`, `&&`, `\|`, `;`, `>` 등 |
| 3. 실행 제한 | `subprocess.run(timeout=10, capture_output=True)` | 10초 타임아웃, 출력 캡처 |

**보안 흐름:**
```
입력 명령어
    │
    ├─ 빈 명령어? → 에러 반환
    ├─ 기본 명령어가 허용 목록에 없음? → 에러 반환
    ├─ 차단 패턴 포함? → 에러 반환
    └─ 통과 → subprocess.run 실행 (타임아웃 10초)
```

**차단 패턴에 셸 연산자(`&&`, `||`, `|`, `;`, `>`, `<`)를 포함**한 이유: `subprocess.run`에 리스트를 전달하면 셸을 거치지 않지만, 만약 `shell=True`로 변경되거나 명령어 인젝션이 시도될 경우를 대비한 방어적 설계이다.

### 3.5 워커 래퍼 함수

래퍼 함수는 부모 그래프와 서브그래프 사이의 **어댑터** 역할을 한다.

```python
def web_search_wrapper(state: State) -> dict:
    # 1. 부모 State.messages → WorkerState.messages로 전달
    result = web_search_subgraph.invoke({"messages": state["messages"]})

    # 2. 서브그래프 결과를 부모 State 형식으로 변환
    # 3. completed_agents 업데이트
    completed = list(state.get("completed_agents", []))
    completed.append("web_search")

    return {
        "messages": [AIMessage(content=f"[웹 검색 결과]\n{last_message.content}")],
        "completed_agents": completed,
    }
```

**왜 서브그래프를 직접 노드로 등록하지 않았는가?**

LangGraph는 컴파일된 그래프를 노드로 직접 등록할 수 있지만, 이 경우 `completed_agents` 같은 부모 State 필드를 서브그래프 내부에서 업데이트할 수 없다. 래퍼 함수를 사용하면 서브그래프 실행과 상태 업데이트를 한 곳에서 처리할 수 있다.

---

## 4. 프로젝트 내 기존 패턴 재사용

본 예제는 기존 예제들에서 검증된 패턴을 조합하여 구현되었다.

| 재사용 패턴 | 출처 | 적용 위치 |
|-------------|------|-----------|
| `@tool` 데코레이터 + `ToolNode` + `bind_tools` | example_6 | 워커 서브그래프의 도구 정의 및 ReAct 루프 |
| `should_continue` 라우터 (tool_calls 유무 판단) | example_6 | `web_search_should_continue`, `cli_should_continue` |
| JSON 파싱 (`extract_json`, 코드 블록 추출) | example_5 | `extract_json_from_text` |
| 파싱 실패 시 안전 폴백 | example_5 | `supervisor_node`에서 `FINISH`로 폴백 |
| 최대 재시도 횟수 제한 | example_5 | `MAX_ITERATIONS = 5` |
| 확장 State (messages 외 커스텀 필드) | example_2 | `next_agent`, `plan`, `completed_agents` 필드 |
| `add_conditional_edges` 다중 타겟 | example_2 | 슈퍼바이저 → {web_search_agent, cli_agent, END} |
| `print` 포맷팅 컨벤션 (`[노드명]` 접두사, 구분선) | example_1 | 모든 노드의 로깅 출력 |

---

## 5. LangGraph 핵심 개념 활용

### 5.1 StateGraph와 서브그래프

- **메인 그래프**: `StateGraph(State)` — 슈퍼바이저와 워커 래퍼를 노드로 등록
- **서브그래프**: `StateGraph(WorkerState)` — 각 워커의 ReAct 루프를 독립적으로 빌드하고 `compile()`

서브그래프는 컴파일된 `CompiledGraph` 객체로, `.invoke()`로 독립 실행이 가능하다.

### 5.2 add_messages 리듀서

`Annotated[list[BaseMessage], add_messages]`는 LangGraph의 핵심 리듀서로, 노드가 반환한 메시지를 기존 리스트에 **추가(append)** 한다. 덮어쓰기가 아닌 누적 방식이므로 대화 이력이 자동으로 보존된다.

### 5.3 조건부 라우팅 (Conditional Edges)

```python
graph.add_conditional_edges(
    "supervisor",           # 소스 노드
    supervisor_router,      # 라우팅 함수 (state → str)
    ["web_search_agent", "cli_agent", END],  # 가능한 타겟 목록
)
```

라우팅 함수가 반환하는 문자열이 타겟 노드 이름과 일치해야 한다. `END`는 그래프 종료를 의미하는 특수 상수이다.

### 5.4 순환 엣지 (Cycles)

```python
graph.add_edge("web_search_agent", "supervisor")
graph.add_edge("cli_agent", "supervisor")
```

워커 → 슈퍼바이저 엣지가 그래프에 **순환(cycle)**을 만든다. 이 순환은 슈퍼바이저가 `FINISH`를 반환하거나 `MAX_ITERATIONS`에 도달할 때 종료된다. LangGraph는 이러한 순환을 공식적으로 지원하며, 조건부 엣지에서 `END`로의 경로가 존재하는 한 유효한 그래프로 컴파일된다.

---

## 6. 실행 시나리오 분석

### 시나리오 A: 웹 검색만

```
입력: "Python 3.12의 새 기능을 검색해주세요"

[START] → [supervisor]
    → JSON: {"next": "web_search", "plan": "1. 웹 검색으로 Python 3.12 새 기능 조사"}
    → [web_search_agent]
        → LLM: web_search("Python 3.12 new features") 호출
        → Tavily 결과 반환
        → LLM: 결과 정리하여 텍스트 응답
    → [supervisor]
    → JSON: {"next": "FINISH", "reason": "검색 완료, 결과 요약..."}
    → [END]
```

### 시나리오 B: CLI만

```
입력: "현재 시스템의 Python 버전과 오늘 날짜를 확인해주세요"

[START] → [supervisor]
    → JSON: {"next": "cli", "plan": "1. CLI로 Python 버전과 날짜 확인"}
    → [cli_agent]
        → LLM: run_cli_command("python3 --version") 호출
        → LLM: run_cli_command("date") 호출
        → LLM: 결과 정리
    → [supervisor]
    → JSON: {"next": "FINISH", "reason": "시스템 정보 확인 완료..."}
    → [END]
```

### 시나리오 C: 복합 요청

```
입력: "Python 3.12 새 기능을 검색하고, 제 시스템의 Python 버전도 확인해주세요"

[START] → [supervisor]
    → JSON: {"next": "web_search", "plan": "1. 웹 검색 2. CLI 확인"}
    → [web_search_agent] (Python 3.12 검색)
    → [supervisor]
    → JSON: {"next": "cli", "plan": "1. 웹 검색 ✓ 2. CLI 확인"}
    → [cli_agent] (python3 --version, date)
    → [supervisor]
    → JSON: {"next": "FINISH", "reason": "모든 작업 완료, 결과 종합..."}
    → [END]
```

---

## 7. 환경 설정

### 필수 환경 변수

| 변수 | 용도 | 발급처 |
|------|------|--------|
| `OPENAI_API_KEY` | 슈퍼바이저 + 워커 LLM 호출 | [OpenAI Platform](https://platform.openai.com/) |
| `TAVILY_API_KEY` | 웹 검색 도구 | [Tavily](https://tavily.com/) |

### 의존성

```
langgraph>=0.2.0
langchain-openai>=0.1.0
langchain-core>=0.2.0
langchain-tavily>=0.1.0
```

---

## 8. 확장 가능성

본 예제는 최소 구성(웹 검색 + CLI)이지만, 아래와 같이 확장할 수 있다.

### 8.1 새 워커 추가 (예: 코드 분석 에이전트)

1. 도구 정의 (`@tool def analyze_code(...)`)
2. 서브그래프 빌드 (`build_code_analyst_agent()`)
3. 래퍼 함수 작성 (`code_analyst_wrapper`)
4. 메인 그래프에 노드 및 엣지 추가
5. 슈퍼바이저 시스템 프롬프트에 새 워커 설명 추가

### 8.2 계층적 멀티 에이전트 (Hierarchical)

슈퍼바이저 자체를 워커로 등록하면 2계층 구조가 가능하다. 예를 들어 "리서치 슈퍼바이저"와 "실행 슈퍼바이저"를 두고, 최상위 슈퍼바이저가 이들을 조율할 수 있다.

### 8.3 병렬 워커 실행

현재는 순차 실행이지만, `Send` API를 사용하면 여러 워커를 동시에 실행할 수 있다. 이 경우 슈퍼바이저의 JSON 출력에 `next`를 리스트로 변경하고, `Send`를 통해 병렬 분기를 만들면 된다.

### 8.4 메모리 및 체크포인팅

`MemorySaver`를 추가하면 대화 이력을 유지하여 멀티턴 대화가 가능하다. example_2의 HITL 패턴과 결합하면 위험한 워커 호출 전 사람의 승인을 받는 시스템도 구현할 수 있다.

---

## 9. 알려진 제약 사항

| 제약 | 설명 | 완화 방안 |
|------|------|-----------|
| 순차 워커 실행 | 한 번에 하나의 워커만 호출 가능 | `Send` API로 병렬화 |
| LLM 모델 고정 | 모든 에이전트가 `gpt-4o-mini` 사용 | 에이전트별 모델 파라미터화 |
| 상태 비지속 | `MemorySaver` 미사용으로 세션 간 상태 유실 | 체크포인터 추가 |
| CLI 보안 | 허용 목록 기반이므로 새 명령어 추가 시 코드 수정 필요 | 설정 파일 기반 허용 목록 |
| 에러 복구 없음 | 워커 실패 시 재시도 로직 없음 | example_5의 자가 교정 루프 적용 |

---

## 10. 참고 자료

### LangGraph 공식 문서
- [Multi-Agent Supervisor 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [서브그래프 가이드](https://langchain-ai.github.io/langgraph/how-tos/subgraph/)
- [멀티 에이전트 협업 패턴](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)
- [계층적 에이전트 팀](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/)

### 도구 API
- [Tavily Search API 문서](https://docs.tavily.com/)
- [langchain-tavily 통합 가이드](https://python.langchain.com/docs/integrations/tools/tavily_search/)
- [Python subprocess 공식 문서](https://docs.python.org/3/library/subprocess.html)

### 프로젝트 내부
- `plan.md` — 전체 프로젝트 아키텍처 설계 원칙
- `examples/example_5_error_handling/main.py` — JSON 파싱, 폴백 패턴
- `examples/example_6_safe_tool_node/main.py` — ReAct 루프, 도구 바인딩 패턴
