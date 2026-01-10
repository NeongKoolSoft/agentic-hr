import os
import re
import streamlit as st

from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -----------------------------
# 0) SQL 정제/검증 유틸
# -----------------------------
def strip_code_fence(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def normalize_sql(s: str) -> str:
    s = strip_code_fence(s)
    s = re.sub(r"^\s*(sql|sqlite)\s*:?\s*", "", s, flags=re.IGNORECASE).strip()

    # SELECT/WITH 시작 전 모든 문자 제거 (ite 같은 찌꺼기 제거)
    m = re.search(r"\b(select|with)\b", s, flags=re.IGNORECASE)
    if m:
        s = s[m.start():].strip()

    s = re.sub(r"\s*```$", "", s).strip()
    return s


def is_safe_readonly_sql(sql: str) -> bool:
    if not sql:
        return False

    lowered = sql.strip().lower()
    parts = [p.strip() for p in lowered.split(";") if p.strip()]
    if len(parts) != 1:
        return False

    forbidden = [
        "insert", "update", "delete", "drop", "alter", "create", "replace",
        "truncate", "attach", "detach", "pragma", "vacuum"
    ]
    if any(tok in lowered for tok in forbidden):
        return False

    return lowered.startswith("select") or lowered.startswith("with")


# -----------------------------
# 1) 페이지 설정 / 사이드바
# -----------------------------
st.set_page_config(page_title="넝쿨 AI 데이터 비서", layout="wide")
st.sidebar.title("⚙️ 설정")

api_key = st.sidebar.text_input("Gemini API Key", type="password", value="")
db_uri = st.sidebar.text_input("DB URI", value="sqlite:///erp_sample.db")
st.sidebar.caption("※ DB URI 예시: sqlite:///erp_sample.db")

if not api_key:
    st.warning("사이드바에 Gemini API 키를 넣어주세요.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# -----------------------------
# 2) DB / LLM 연결
# -----------------------------
db = SQLDatabase.from_uri(db_uri)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
execute_query = QuerySQLDatabaseTool(db=db)
schema_text = db.get_table_info()


# -----------------------------
# 3) SQL 생성 / 답변 프롬프트
# -----------------------------
sql_prompt = PromptTemplate.from_template(
    """You are a {dialect} SQL expert.
Write ONE SQLite query that answers the question using the schema.

Rules:
- Return ONLY SQL (no explanation).
- Do NOT use markdown code fences.
- Read-only SELECT/CTE only. Never use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.
- Prefer "ORDER BY ... DESC LIMIT 1" for "top/highest" questions.
- Use only tables/columns that exist in the schema.

Schema:
{schema}

Question:
{question}

SQL:
"""
)

write_query = (
    {
        "dialect": lambda _: db.dialect,
        "schema": lambda _: schema_text,
        "question": RunnablePassthrough(),
    }
    | sql_prompt
    | llm
    | StrOutputParser()
)

answer_prompt = PromptTemplate.from_template(
    """주어진 질문, SQL 쿼리, 그리고 SQL 결과를 바탕으로 사용자의 질문에 친절하게 답하세요.
질문: {question}
SQL 쿼리: {query}
SQL 결과: {result}
답변:"""
)


# -----------------------------
# 4) 질문 실행 함수
# -----------------------------
def run_question(user_question: str) -> str:
    raw_sql = write_query.invoke(user_question)
    generated_sql = normalize_sql(raw_sql)

    if not is_safe_readonly_sql(generated_sql):
        st.error("안전상 이유로 실행할 수 없는 SQL이 생성되었습니다. 질문을 조금 더 구체적으로 해주세요.")
        with st.expander("🔍 생성된 SQL(실행 차단)"):
            st.code(generated_sql, language="sql")
        return "죄송합니다. 안전을 위해 해당 요청은 실행할 수 없습니다."

    try:
        result = execute_query.invoke({"query": generated_sql})
    except Exception as e:
        result = f"Error: {e}"

    response_text = (
        {"question": lambda _: user_question,
         "query": lambda _: generated_sql,
         "result": lambda _: result}
        | answer_prompt
        | llm
        | StrOutputParser()
    ).invoke({})

    st.markdown(response_text)
    with st.expander("🔍 실행된 SQL 쿼리 확인"):
        st.code(generated_sql, language="sql")

    return response_text


# -----------------------------
# 5) 메인 UI
# -----------------------------
st.title("🌿 넝쿨 AI 데이터 에이전트")
st.subheader("ERP 데이터를 자연어로 조회하세요")

# 세션 상태
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ✅ 대표 질문 10개 (칩)
st.markdown("#### 💡 대표 질문 10개")
chip_questions = [
    "현재 등록된 상품들 중 매출액이 가장 큰 것은 무엇이고 금액은 얼마야?",
    "이번 달(또는 최근 기간) 매출 합계는 얼마야?",
    "상품별 매출 TOP 5를 보여줘.",
    "매출이 0원이거나 거의 없는 상품이 있어?",
    "최근 7일 동안 가장 많이 팔린 상품은 뭐야?",
    "특정 상품(예: 넝쿨OS Pro)의 매출 추이를 보여줘.",
    "오늘 기준 미수금(또는 외상)이 있는 거래처가 있어?",
    "거래처별 매출 TOP 5를 알려줘.",
    "재고가 부족한(예: 10개 이하) 상품 목록을 보여줘.",
    "지난달 대비 이번 달 매출이 얼마나 증가/감소했어?",
]

# 칩을 한 줄에 너무 많이 붙이면 깨져서 5개씩 나눔
rows = [chip_questions[i:i+5] for i in range(0, len(chip_questions), 5)]
for row in rows:
    cols = st.columns(len(row))
    for i, q in enumerate(row):
        if cols[i].button(q, use_container_width=True):
            st.session_state.pending_prompt = q
            st.rerun()

st.divider()

# 기존 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ 칩 클릭으로 들어온 질문 처리
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("DB 분석 중..."):
            response = run_question(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})

# 사용자 직접 입력
user_input = st.chat_input("예: 가장 매출이 높은 상품은 뭐야?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("DB 분석 중..."):
            response = run_question(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
