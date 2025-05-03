from main.prompt import prompt

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import openai

# Ollama 또는 OpenAI 등 원하는 LLM 설정
llama_3_1_8B_instrcut = ChatOllama(model="llama-3.1-8B-instrcut:latest", top_p=0.95, num_predict=200)
mistral_7b = ChatOllama(model="mistral-7b:latest", top_p=0.95, num_predict=200)


def ollama_responses(linearized_table, llm_name) -> str:
    prompt = ChatPromptTemplate.from_template("{prompt}")

    chain = prompt | llm_name | StrOutputParser()

    user_prompt = prompt(linearized_table)

    response = chain.invoke({"prompt": user_prompt})
    return response.strip()

def gpt_responses(linearized_table: str) -> str:
    
    prompt = prompt(linearized_table)

    response = openai.ChatCompletion.create(
        model="gpt4o-mini",
        messages=[
            {"role": "system", "content": "당신은 데이터를 분석하고 요약하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
    )

    return response.choices[0].message['content'].strip()