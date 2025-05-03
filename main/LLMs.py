import os
import openai
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prompt import prompting

'''
input: A linearized table
output: LLM Response
'''

# .env 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ollama_responses(prompt, llm_model) -> str:
    prompt_template = ChatPromptTemplate.from_template("{prompt}")

    chain = prompt_template | llm_model | StrOutputParser()

    user_prompt = prompt

    response = chain.invoke({"prompt": user_prompt})
    return response.strip()


def gpt_responses(prompt, llm_model = None) -> str:

    user_prompt = prompt

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