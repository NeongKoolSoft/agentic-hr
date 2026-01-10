import os
import re
import streamlit as st
from typing import Any, Dict, List, Optional

from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 0) 유틸: SQL 정제
# -----------------------------
def strip_code_fence(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def normalize_sql(sql: str) -> str:
    """
    LLM이 가끔 뱉는 쓰레기 토큰(ite, sql 등) / 코드펜스 / 이상한 접두어 제거
    """
    sql = strip_code_fence(sql)

    # 흔한 쓰레기 토큰 패턴 제거(필요시 추가)
    sql = re.sub(r"^\s*(?:ite|sql)\s*", "", sql, flags=re.IGNORECASE).strip()

    # 세미콜론 정리
    sql = sql.strip().rstrip(";").strip() + ";"
    return sql

def is_safe_select_only(sql: str) -> bool:
    """
    아주 단순한 안전장치: SELECT만 허용 (필요하면 더 강화)
    """
    s = (sql or "").strip().lower()
    # 주석 제거(간단)
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE).strip()

    blocked = ["insert", "update", "delete", "drop", "alter", "create", "replace", "truncate", "attach", "pragma"]
    if not s.startswith("select"):
        return False
    if any(b in s for b in blocked):
        return False
    return True


# -----------------------------
# 1) 엔진 구성(예시): Text-to-SQL + 결과 해석
# -----------------------------
SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior data analyst. Generate a SINGLE SQLite SELECT query.\n"
     "- Only SELECT queries.\n"
     "- Use correct table/column names.\n"
     "- Return ONLY the SQL, no explanations."),
    ("human",
     "User question: {question}\n\nDatabase schema:\n{schema}\n\nSQL:")
])

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an ERP data assistant. Explain the SQL result in Korean concisely.\n"
     "- If the result is empty, say so and suggest a next question.\n"
     "- Include key numbers with currency formatting when relevant."),
    ("human",
     "User question: {question}\nSQL executed:\n{sql}\n\nResult:\n{result}\n\nAnswer:")
])

def build_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0
    )

def get_db(db_uri: str) -> SQLDatabase:
    return SQLDatabase.from_uri(db_uri)

def generate_sql(llm, schema: str, question: str) -> str:
    chain = SQL_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"schema": schema, "question": question})
    return normalize_sql(raw)

def run_sql(db: SQLDatabase, sql: str) -> str:
    """
    db.run()은 결과를 문자열로 반환(드라이버에 따라 다름).
    너가 이미 쓰던 실행 로직이 있으면 여기만 교체하면 됨.
    """
    return db.run(sql)

def explain_result(llm, question: str, sql: str, result: str) -> str:
    chain = EXPLAIN_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question, "sql": sql, "result": result})


# -----------------------------
# 2) Streamlit UI
# -----------------------------
st.set_page_config(page_title="넝쿨 AI 데이터 에이전트", layout="wide")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant", "content": "...", "sql": "..."}]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Sidebar 설정
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    db_uri = st.text_input("DB URI", value="sqlite:///erp_sample.db")
    st.caption("※ DB URI 예시: sqlite:///erp_sample.db")

# 헤더
st.markdown(
    """
    <div style="text-align:center; padding-top:12px;">
      <h1>🌿 넝쿨 AI 데이터 에이전트</h1>
      <h3 style="font-weight:600;">ERP 데이터를 자연어로 조회하세요</h3>
    </div>
    """,
    unsafe_allow_html=True
)

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

# 버튼을 "칩 느낌"으로: 여러 열로 나눠 배치
cols = st.columns(2)
for i, q in enumerate(chip_questions):
    with cols[i % 2]:
        if st.button(q, use_container_width=True, key=f"chip_{i}"):
            st.session_state.pending_question = q
            st.rerun()

st.divider()

# 기존 메시지 렌더링
for m in st.session_state.messages:
    if m["role"] == "user":
        st.chat_message("user").write(m["content"])
    else:
        with st.chat_message("assistant"):
            st.write(m["content"])
            if m.get("sql"):
                with st.expander("🔎 실행된 SQL 보기"):
                    st.code(m["sql"], language="sql")
                    

# 채팅 입력
user_text = st.chat_input("예: 현재 등록된 상품들 중 매출액이 가장 큰 것은 뭐야?")

# 칩 클릭 질문 우선 처리
question = None
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
elif user_text:
    question = user_text

# 실행 로직 (공통)
if question:
    if not api_key:
        st.warning("Gemini API Key를 입력해줘.")
        st.stop()
    if not db_uri:
        st.warning("DB URI를 입력해줘.")
        st.stop()

    # 사용자 메시지 저장/표시
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    try:
        llm = build_llm(api_key)
        db = get_db(db_uri)

        # 스키마
        schema = db.get_table_info()

        # SQL 생성
        sql = generate_sql(llm, schema=schema, question=question)

        # 안전 검사
        if not is_safe_select_only(sql):
            raise ValueError("안전 정책상 SELECT 쿼리만 허용됩니다. 생성된 SQL이 차단되었습니다.")

        # DB 실행
        result = run_sql(db, sql)

        # 결과 해석
        answer = explain_result(llm, question=question, sql=sql, result=result)

        # 어시스턴트 메시지 저장/표시 + SQL Expander
        st.session_state.messages.append({"role": "assistant", "content": answer, "sql": sql})
        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("🔎 실행된 SQL 보기"):
                st.text_area("SQL", sql, height=120, label_visibility="collapsed")

    except Exception as e:
        err_msg = f"에러가 발생했어: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
        st.chat_message("assistant").error(err_msg)
