import pandas as pd

# 각 행을 하나의 문장으로 변환, 하나의 행이 끝나면 | 로 분리 - Horizontal
def linearize_row_wise(df):
    return " | ".join(["; ".join([f"{col}: {val}" for col, val in row.items()]) for _, row in df.iterrows()])

# 각 열별로 내용들을 이어 붙여 문장으로 변환, 각 열의 내용들을 쭉 나열하고 하나의 열이 끝나면 | 로 분리 - Vertical
def linearize_column_wise(df):
    return " | ".join([f"{col}: {', '.join(map(str, df[col].tolist()))}" for col in df.columns])

# 각 행들을 "column 명 : cell 값" pair로 구성하여 행별로 제공, 하나의 행이 끝나면 | 로 분리
def linearize_flatten_schema(df):
    rows = []
    for _, row in df.iterrows():
        rows.append(" ".join([f"{col}: {val}" for col, val in row.items()]))
    return " [SEP] ".join(rows)

# table을 markdown 형태의 텍스트로 변형하여 table처럼 보이도록 구성
def linearize_markdown(df):
    md = "| " + " | ".join(df.columns) + " |\n"
    md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join(map(str, row)) + " |\n"
    return md

# table 속 내용을 json 형태로 변형하여 제공
def linearize_json(df):
    return df.to_json(orient='records')

# 열의 이름과 해당 셀 값을 "열 이름은 값입니다"로 자연스럽게 연결
def linearize_natural_language(df):
    sents = []
    for _, row in df.iterrows():
        sent = ", ".join([f"{col}은(는) {val}입니다" for col, val in row.items()])
        sents.append(sent)
    return " [SEP] ".join(sents)

# row_wise = linearize_row_wise(df)
# column_wise = linearize_column_wise(df)
# flatten_schema = linearize_flatten_schema(df)
# markdown = linearize_markdown(df)
# json_type = linearize_json(df)
# natural_language = linearize_natural_language(df)