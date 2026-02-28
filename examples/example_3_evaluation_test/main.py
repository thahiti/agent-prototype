"""
예제 3: 통합 테스트 — LLM-as-a-Judge & 임베딩 유사도 평가

핵심 아이디어:
- 에이전트의 출력 품질을 자동으로 검증하는 두 가지 평가기를 구현한다.
  1) 임베딩 유사도 평가: 모델 출력과 정답 사이의 코사인 유사도를 측정.
  2) LLM 심판 평가: 별도의 LLM이 1~5점으로 정확도를 채점.
- CI/CD에서 pytest로 실행하여 품질 기준치를 자동으로 검증할 수 있다.

실행 방법:
    uv run examples/example_3_evaluation_test/main.py

구조:
    [평가 대상 그래프] → 출력 생성 → [평가기 1: 임베딩 유사도]
                                    → [평가기 2: LLM 심판]
"""

import json
import numpy as np
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1단계: 평가 대상이 될 간단한 에이전트 그래프
# ============================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state: State) -> dict:
    """평가 대상이 되는 챗봇 노드."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_agent_graph():
    """평가 대상 그래프: START → chatbot → END"""
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    return graph.compile()


# ============================================================
# 2단계: 평가기 A — 임베딩 코사인 유사도
# ============================================================
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """두 벡터 사이의 코사인 유사도를 계산한다.

    공식: cos(θ) = (A · B) / (||A|| × ||B||)
    - 1.0에 가까울수록 의미적으로 유사
    - 0.0이면 관련 없음
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    # 분모가 0인 경우(제로 벡터) 방어
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        return 0.0
    return float(np.dot(a, b) / norm_product)


def evaluate_embedding_similarity(
    model_output: str,
    reference: str,
    threshold: float = 0.85,
) -> dict:
    """모델 출력과 정답의 임베딩 유사도를 평가한다.

    Args:
        model_output: 에이전트가 생성한 답변
        reference: 기대하는 정답
        threshold: 통과 기준치 (기본 0.85)

    Returns:
        {
            "similarity": 0.92,
            "threshold": 0.85,
            "passed": True
        }
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 두 텍스트를 동시에 임베딩
    vectors = embeddings.embed_documents([model_output, reference])
    output_vec = vectors[0]
    reference_vec = vectors[1]

    # 코사인 유사도 계산
    similarity = cosine_similarity(output_vec, reference_vec)

    result = {
        "similarity": round(similarity, 4),
        "threshold": threshold,
        "passed": similarity >= threshold,
    }

    print(f"[임베딩 평가] 유사도: {result['similarity']} "
          f"(기준: {threshold}) → {'PASS ✅' if result['passed'] else 'FAIL ❌'}")

    return result


# ============================================================
# 3단계: 평가기 B — LLM-as-a-Judge (LLM 심판)
# ============================================================
def evaluate_llm_judge(
    question: str,
    model_output: str,
    reference: str,
    passing_score: int = 4,
) -> dict:
    """독립적인 LLM이 모델 출력의 정확도를 1~5점으로 채점한다.

    Args:
        question: 원래 질문
        model_output: 에이전트의 답변
        reference: 기대하는 정답
        passing_score: 통과 기준 점수 (기본 4점)

    Returns:
        {
            "score": 5,
            "reasoning": "...",
            "passed": True
        }
    """
    # 심판 역할을 하는 별도의 LLM
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 평가 프롬프트: 명확한 기준과 JSON 출력 형식을 지정
    judge_prompt = f"""당신은 AI 답변 품질 평가 심판입니다.
아래 정보를 바탕으로 모델의 답변이 얼마나 정확한지 1~5점으로 평가하세요.

## 채점 기준
- 5점: 정답과 동일한 수준의 정확한 답변
- 4점: 핵심 내용은 맞지만 세부사항이 약간 다름
- 3점: 부분적으로 맞지만 중요한 내용이 빠짐
- 2점: 대부분 틀리거나 관련 없는 내용
- 1점: 완전히 틀린 답변

## 입력
- 질문: {question}
- 정답(Reference): {reference}
- 모델 답변: {model_output}

## 출력 형식 (반드시 JSON)
{{"score": <1-5>, "reasoning": "<평가 이유>"}}
"""

    response = judge_llm.invoke([HumanMessage(content=judge_prompt)])

    # JSON 파싱
    parsed = json.loads(response.content)
    score = parsed["score"]
    reasoning = parsed["reasoning"]

    result = {
        "score": score,
        "reasoning": reasoning,
        "passed": score >= passing_score,
    }

    print(f"[LLM 심판] 점수: {score}/5 → {'PASS ✅' if result['passed'] else 'FAIL ❌'}")
    print(f"[LLM 심판] 이유: {reasoning}")

    return result


# ============================================================
# 4단계: 통합 테스트 실행
# ============================================================
# 테스트 케이스 정의: (질문, 기대 정답)
TEST_CASES = [
    {
        "question": "대한민국의 수도는 어디인가요?",
        "reference": "대한민국의 수도는 서울입니다.",
    },
    {
        "question": "Python에서 리스트의 길이를 구하는 함수는?",
        "reference": "len() 함수를 사용하여 리스트의 길이를 구할 수 있습니다.",
    },
]


def main():
    app = build_agent_graph()

    print("=" * 60)
    print("예제 3: 통합 테스트 (임베딩 유사도 + LLM 심판) 데모")
    print("=" * 60)

    all_passed = True

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'─'*60}")
        print(f"테스트 케이스 #{i}: {test_case['question']}")
        print(f"{'─'*60}")

        # 에이전트 실행
        result = app.invoke({
            "messages": [HumanMessage(content=test_case["question"])]
        })
        model_output = result["messages"][-1].content
        print(f"모델 답변: {model_output[:100]}...")

        # 평가기 A: 임베딩 유사도
        embed_result = evaluate_embedding_similarity(
            model_output=model_output,
            reference=test_case["reference"],
            threshold=0.80,  # 데모용으로 기준을 살짝 낮춤
        )

        # 평가기 B: LLM 심판
        judge_result = evaluate_llm_judge(
            question=test_case["question"],
            model_output=model_output,
            reference=test_case["reference"],
            passing_score=4,
        )

        # 두 평가기 모두 통과해야 테스트 통과
        case_passed = embed_result["passed"] and judge_result["passed"]
        if not case_passed:
            all_passed = False

        print(f"\n📋 케이스 #{i} 종합: {'PASS ✅' if case_passed else 'FAIL ❌'}")

    print(f"\n{'='*60}")
    print(f"전체 결과: {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
