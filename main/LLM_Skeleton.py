from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.chat_models import ChatOllama
from textwrap import dedent

from src.table_linearlization import linearize_row_wise
from src.table_parser import load_survey_tables
from src.table_numeric_analysis import main as numeric_analysis


def get_user_selected_table(file_path):
    tables, question_texts, question_keys = load_survey_tables(file_path)

    print("Question key and texts list:")
    for i, key in enumerate(question_keys):
        print(f"{i+1}. [{key}] {question_texts[key]}")

    user_input = input("\n요약을 생성할 질문 번호(예: 1) 또는 질문 키(예: A1, B14_2)를 입력하세요: ").strip()

    if user_input.isdigit():
        choice = int(user_input) - 1
        if choice < 0 or choice >= len(question_keys):
            raise ValueError("올바른 질문 번호를 입력하세요.")
        selected_key = question_keys[choice]
    elif user_input in question_keys:
        selected_key = user_input
    else:
        raise ValueError("입력한 값이 올바른 질문 번호 또는 키가 아닙니다.")

    return tables[selected_key], question_texts[selected_key], selected_key


def get_skeleton_template():
    return PromptTemplate(
        input_variables=["linearized_table"],
        template=dedent("""
        당신은 통계 요약 전문가입니다. 아래 설문조사 표 데이터를 보고 요약에 반드시 포함되어야 할 핵심 정보를 추출하세요.

        📊 [표 데이터]
        {linearized_table}

        다음 조건에 따라 3~6개의 핵심 요점을 문장으로 나열하세요:
        - 평균, 극단값, 집단 간 차이, 표준편차가 큰 항목
        - 요약에 반드시 들어가야 할 수치 기반 정보

        출력 형식 예시:
        - 60대는 평균 4.1로 가장 높은 관심도를 보임
        - 실외 체류자는 '매우 관심 있다' 비율이 25.9%로 가장 높음
        - 기저질환자 보유 여부에 따라 '관심 있다' 비율이 11% 차이남
        """)
    )


def get_summary_template():
    return PromptTemplate(
        input_variables=["linearized_table", "skeleton", "analysis_results"],
        template=dedent("""
        당신은 객관적인 통계 기반 요약을 전문으로 하는 분석가입니다.
        아래는 설문조사 데이터를 정리한 표이며, 각 행은 인구집단(예: 성별, 연령대), 각 열은 해당 집단의 응답 통계를 나타냅니다.

        📊 [표 데이터 (선형화된 형태)]
        {linearized_table}

        📈 [수치 분석 결과 (대분류별 항목별 max/min 및 분산 등)]
        {analysis_results}

        📝 [참고용 스켈레톤 요점들]
        다음은 통계적으로 주목할 만한 응답 경향을 요약한 예시입니다. 반드시 포함할 필요는 없지만, 요약 작성에 참고해도 좋습니다.
        {skeleton}

        🧠 Chain of Thought 추론 순서:
        1. 표와 수치 분석 결과를 종합적으로 검토하여 의미 있는 패턴을 식별합니다.
        2. 평균, 극단값, 표준편차 등 수치 기반의 차이를 중심으로 해석합니다.
        3. 참고 요점과 비교하여 누락된 중요한 통계적 특징이 있다면 반영합니다.

        📝 작성 조건:
        - 반드시 수치 기반 근거만을 바탕으로 해석하며, 외부 배경지식은 사용하지 마세요.
        - 모든 소분류를 나열하지 마세요. 유의미한 차이를 보이는 항목 위주로 간결하게 작성하세요.
        - 문장은 **객관적이고 명확하게**, **서술형 문단 1~2개**로 요약해 주세요.
        - 제목은 쓰지 마시고, 요약 내용만 출력하세요.
        """)
    )


def run_summary_pipeline(table, linearized_table, analysis_results, model_name="llama-3.1-8B-instrcut:latest"):
    llm = ChatOllama(model=model_name, temperature=0.3)

    skeleton_chain = LLMChain(llm=llm, prompt=get_skeleton_template(), output_key="skeleton")
    summary_chain = LLMChain(llm=llm, prompt=get_summary_template(), output_key="summary")

    full_chain = SequentialChain(
        chains=[skeleton_chain, summary_chain],
        input_variables=["linearized_table", "analysis_results"],
        output_variables=["skeleton", "summary"],
        verbose=True
    )

    return full_chain.invoke({
        "linearized_table": linearized_table,
        "analysis_results": analysis_results
    })


def main():
    file_path = "통계표_수정_서울시 대기환경 시민인식 조사_250421(작업용).xlsx"

    table, question_text, key = get_user_selected_table(file_path)
    linearized = linearize_row_wise(table)
    analysis = numeric_analysis(table)

    result = run_summary_pipeline(table, linearized, analysis)

    print("\n🔑 [스켈레톤 요점]")
    print(result["skeleton"])
    print("-" * 20)

    print("\n🔑 [수치 분석 결과]")
    print(analysis)
    print("-" * 20)

    print(f"\n📝 [최종 요약 for {key}]")
    print(result["summary"])


if __name__ == "__main__":
    main()