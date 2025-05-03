import os
import openai
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

'''
input: A linearized table
output: LLM Response
'''

# .env 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def prompting(linearized_table, analysis_results=None):
    prompt = (
        "당신은 객관적인 통계 기반 요약을 전문으로 하는 분석가입니다.\n"
        "아래는 서울시 대기환경 관련 설문조사 데이터를 정리한 표이며, 각 행은 인구집단(예: 성별, 연령대), 각 열은 해당 집단의 응답 통계를 나타냅니다.\n\n"
        f"📊 [표 데이터 (선형화된 형태)]\n{linearized_table}\n\n"
        f"📈 [수치 분석 결과 (대분류별 항목별 max/min 및 분산 등)]\n{analysis_results}\n\n"
        "🧠 Chain of Thought 추론 순서:\n"
        "1. 각 항목(예: 관심도 평균, 관심있다 비율 등)에 대해 그룹 간 수치 차이가 뚜렷한지 파악합니다.\n"
        "2. 극단값(최고/최저 그룹)과 평균, 표준편차 또는 범위(range)를 비교하여 의미 있는 차이를 가진 대분류 그룹을 우선적으로 분석합니다.\n"
        "3. 도출된 특이점(예: 60대는 평균 4.1로 가장 높음)을 바탕으로 간결하고 객관적인 요약문을 작성합니다.\n\n"
        "📝 작성 조건:\n"
        "- 수치 기반 특징(예: 평균, % 차이 등)만을 바탕으로 해석하며, 외부 배경지식은 사용하지 마세요.\n"
        "- 모든 소분류를 열거하지 마세요. 변동폭이 크거나 특징적인 그룹만 선택적으로 언급하세요.\n"
        "- 문장은 **간결하고 명확하며**, 주관적 표현 없이 **객관적인 문체**로 작성해 주세요.\n"
        "- 한국어로 서술형 단락 1~2개로 구성된 보고서를 작성하세요.\n"
        "- 보고서에는 문단 제목 없이, 요약 내용만 포함합니다.\n\n"
        "🧾 최종 출력 예시 (형식):\n"
        "대기환경에 대한 관심도는 연령, 건강상태, 주요 체류 공간에 따라 유의미한 차이를 보였다. 특히 60대 이상은 평균 4.1점으로 다른 연령대보다 높은 관심을 보였으며, 기저질환자는 ‘매우 관심 있다’ 비율이 상대적으로 높았다. 실외 활동이 많은 그룹 역시 관심도 평균이 4.0으로 높은 편이었다."
    )
    return prompt

def ollama_responses(linearized_table, analysis_results, llm_model) -> str:
    prompt_template = ChatPromptTemplate.from_template("{prompt}")

    chain = prompt_template | llm_model | StrOutputParser()

    user_prompt = prompting(linearized_table, analysis_results)

    response = chain.invoke({"prompt": user_prompt})
    return response.strip()


def gpt_responses(linearized_table, analysis_results) -> str:

    user_prompt = prompting(linearized_table, analysis_results)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 데이터를 분석하고 요약하는 전문가입니다."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=1500,
    )

    return response.choices[0].message['content'].strip()