# 📊 Table_Exercise: 설문조사 표 자동 요약 프로젝트

이 프로젝트는 엑셀 형식으로 제공되는 설문조사 결과 테이블을 자동으로 전처리하고, LLM을 활용하여 자연어 요약 문장을 생성하는 파이프라인을 제공합니다.

---

## 📌 주요 기능

1. **엑셀 파일 자동 파싱**
   - 질문 항목(A1., A2. 등)을 기준으로 각 설문 테이블 자동 분할
   - 병합된 컬럼 헤더 구성
   - 전체 요약 및 기타/무응답 행 제거
   - 숫자 값은 소수점 1자리로 반올림

2. **테이블 Linearization**
   - 테이블을 다양한 포맷(자연어, 마크다운, JSON 등)으로 변환하여 LLM이 이해할 수 있도록 정제

3. **LLM 기반 문장 생성**
   - 변환된 테이블 데이터를 LLM에게 전달하여 설문 결과 요약문 자동 생성
   - OpenAI API 또는 Ollama LLaMA3 모델 사용 가능

---

## 📁 디렉토리 구조
Table_Exercise/
├── data/                    # 엑셀 데이터 (Git 업로드 제외)
├── research_AI/
│   ├── preprocess.py        # 테이블 전처리 및 분할 함수
│   ├── linearizers.py       # 테이블 → 텍스트 변환 함수
│   ├── generate_summary.py  # LLM 호출 및 요약 생성 함수
│   └── README.md            # 프로젝트 설명서
├── api/
│   └── prompts.py           # 다양한 LLM용 프롬프트 정의
├── .gitignore
└── requirements.txt

---

## 🚀 실행 방법

```bash
# 가상환경 설정 (선택)
python -m venv venv
source venv/bin/activate

# 라이브러리 설치
pip install -r requirements.txt

# 엑셀 테이블 전처리 후 요약 생성 예시
python research_AI/generate_summary.py