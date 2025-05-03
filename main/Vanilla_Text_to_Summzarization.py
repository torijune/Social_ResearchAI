from src.table_parser import load_survey_tables

from src.table_linearlization import linearize_row_wise, linearize_column_wise
from src.table_linearlization import linearize_flatten_schema, linearize_markdown, linearize_json, linearize_natural_language

from main.LLMs import ollama_responses, gpt_responses
from main.prompt import prompting

from langchain_ollama import ChatOllama

'''
Input: table -> linearized_table -> prompting -> LLMs -> Output: LLM Respones
'''

def Vanilla_Text_to_Summzarization(
    file_path: str,
    model_name: str = "llama-3.1-8B-instrcut:latest",
    # table flatten 방법론 설정
    linearization_fn=linearize_row_wise,
    # LLM 설정
    response_fn=ollama_responses, # gpt_responses로 변경 가능
    top_p: float = 0.95,
    num_predict: int = 1500
):
    """
    엑셀 설문 데이터를 로드하고, 사용자가 선택한 질문에 대해 요약을 생성하는 baseline 함수.

    Args:
        file_path (str): 엑셀 파일 경로
        model_name (str): 사용할 Ollama 모델 이름
        linearization_fn (callable): 테이블을 문자열로 변환하는 함수
        response_fn (callable): LLM 응답 생성 함수
        top_p (float): sampling top_p 설정 (Ollama)
        num_predict (int): 예측 최대 토큰 수

    Returns:
        dict: {
            "selected_key": str,
            "question": str,
            "summary": str
        }
    """

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

    selected_table = tables[selected_key]
    selected_question = question_texts[selected_key]

    # Table → linearized
    linearized = linearization_fn(selected_table)

    # LLM 인스턴스 생성
    model = ChatOllama(model=model_name, top_p=top_p, num_predict=num_predict)

    prompt = prompting(linearized)
    # 요약 생성
    summary = response_fn(prompt, model)

    print(f"\n✅ 요약 결과 for [{selected_key}]: {selected_question}\n")
    print(summary)

    return {
        "selected_key": selected_key,
        "question": selected_question,
        "summary": summary
    }

if __name__ == "__main__":
    file_path = "통계표_수정_서울시 대기환경 시민인식 조사_250421(작업용).xlsx"
    result = Vanilla_Text_to_Summzarization(file_path)