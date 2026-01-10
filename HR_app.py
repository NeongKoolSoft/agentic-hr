# HR_app.py
import uuid
import streamlit as st

from HR_sql_ai import HRTextToSQLEngine, ENGINE_VERSION
from scenario_payroll import ScenarioMemoryManager, PayrollScenario, ScenarioOrchestrator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# 유틸: 메시지 → 턴 구조
# =====================================================
def build_turns(messages):
    turns = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if m["role"] == "user":
            turn = {"user": m, "assistant": None}
            if i + 1 < n and messages[i + 1]["role"] == "assistant":
                turn["assistant"] = messages[i + 1]
                i += 2
            else:
                i += 1
            turns.append(turn)
        else:
            turns.append({"user": None, "assistant": m})
            i += 1
    return turns


def render_action_chips(suggestions, key_prefix="act"):
    """시나리오가 제안하는 다음 행동(예/아니오/지급 진행 등)을 버튼 칩으로 렌더링"""
    if not suggestions:
        return None

    shown = suggestions[:4]
    cols = st.columns(len(shown))
    for i, label in enumerate(shown):
        if cols[i].button(label, key=f"{key_prefix}_{i}_{label}", use_container_width=True):
            return label
    return None


# =====================================================
# CSS (상단 공백 제거 + 중앙 로딩 오버레이)
# =====================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 1rem;
    }
    @media (max-width: 768px) {
        .block-container { padding-top: 0.4rem !important; }
    }

    .nk-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.08);
        z-index: 9998;
    }
    .nk-center-spinner {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        background: rgba(255,255,255,0.96);
        padding: 26px 34px;
        border-radius: 14px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        text-align: center;
        font-size: 16px;
        font-weight: 700;
        min-width: 260px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def show_center_spinner(text: str = "처리 중..."):
    return st.markdown(
        f"""
        <div class="nk-overlay"></div>
        <div class="nk-center-spinner">⏳ {text}</div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# 1) 페이지 설정 / 세션
# =====================================================
st.set_page_config(page_title="Agentic AI for 넝쿨HR", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ✅ 시나리오 다음 작업(칩) 저장소
if "action_suggestions" not in st.session_state:
    st.session_state.action_suggestions = []


# =====================================================
# 2) secrets 로드
# =====================================================
api_key = st.secrets.get("GOOGLE_API_KEY")
db_uri = st.secrets.get("SUPABASE_DB_URI")

if not api_key or not db_uri:
    st.error(
        "❌ 환경 설정이 없습니다.\n\n"
        ".streamlit/secrets.toml 에 다음 값을 설정하세요:\n"
        "- GOOGLE_API_KEY\n"
        "- SUPABASE_DB_URI"
    )
    st.stop()

if "[YOUR-PASSWORD]" in db_uri:
    st.error("❌ SUPABASE_DB_URI에 [YOUR-PASSWORD]가 그대로 있습니다.")
    st.stop()


# =====================================================
# 3) 엔진 / 설명기 (캐시는 유지)
# =====================================================
@st.cache_resource(show_spinner=False)
def get_engine(_db_uri: str, _api_key: str, _version: str) -> HRTextToSQLEngine:
    return HRTextToSQLEngine(db_uri=_db_uri, api_key=_api_key)


@st.cache_resource(show_spinner=False)
def get_explainer(_api_key: str):
    prompt = ChatPromptTemplate.from_template(
        """다음은 HR 데이터 조회 결과입니다.
질문: {question}
SQL 결과: {result}

한국어로 간결하게 설명하세요.
- 결과가 없으면 '해당 조건에 데이터가 없습니다'라고 답하세요.
- 오류가 있으면 원인을 짧게 요약하고 다음 행동을 제안하세요."""
    )
    return (
        prompt
        | ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=_api_key,
            temperature=0
        )
        | StrOutputParser()
    )


engine = get_engine(db_uri, api_key, ENGINE_VERSION)
explainer = get_explainer(api_key)

# ============================
# 시나리오 메모리 / 오케스트레이터
# ============================
memory = ScenarioMemoryManager(store=st.session_state, namespace="scenario_memory")
payroll_scenario = PayrollScenario(memory)

orchestrator = ScenarioOrchestrator(
    sql_engine=engine,
    scenarios=[payroll_scenario],
)


# =====================================================
# 4) 헤더 (공간 줄임)
# =====================================================
st.markdown(
    """
    <div style="text-align:center; padding:15px 0 4px 0;">
      <h2 style="margin:0;">Agentic AI for 넝쿨HR</h2>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 🧭 시나리오 상태 바 + 종료
# =====================================================
STATE_LABEL = {
    "PAYROLL_CALC": "급여 산정",
    "TAX_CALC": "공제 검증",
    "PAYMENT_RUN": "지급 처리",
    "JOURNAL_POST": "전표 생성",
    "DONE": "완료",
}

ctx = st.session_state.get("scenario_memory", {}).get(st.session_state.session_id)

if ctx and ctx.get("active_scenario"):
    state = ctx.get("state")
    st.info(f"🧭 현재 작업: 급여 처리 · 단계: {STATE_LABEL.get(state, state)}")

    if st.button("❌ 시나리오 종료"):
        memory.clear(st.session_state.session_id)
        st.session_state.action_suggestions = []
        st.success("시나리오가 종료되었습니다.")
        st.rerun()


# =====================================================
# 5) 대표 질문
# =====================================================
chip_questions = [
    "부서별 재직 인원수는?",
    "최근 30일 신규 입사자는 누구야?",
    "최근 90일 퇴사자는 누구야?",
    "이번 주 지각이나 결근이 가장 많은 직원 TOP 5는?",
    "이번 달 총 근무시간 TOP 5는?",
    "이번 달 부서별 평균 근무시간은?",
    "이번 달 승인된 휴가 사용일수 TOP 5는?",
    "현재 승인 대기 중인 휴가 요청은?",
    "2025년 12월 직급별 평균 실수령은?",
    "2025년 12월 부서별 실수령 총액은?",
]

cols = st.columns(2)
for i, q in enumerate(chip_questions):
    with cols[i % 2]:
        if st.button(q, use_container_width=True, key=f"chip_{i}"):
            st.session_state.pending_question = q
            st.rerun()

st.divider()


# =====================================================
# 6) 기존 대화 표시
# =====================================================
turns = build_turns(st.session_state.messages)

for t in reversed(turns):
    if t["user"]:
        with st.chat_message("user"):
            st.markdown(t["user"]["content"])

    if t["assistant"]:
        with st.chat_message("assistant"):
            st.markdown(t["assistant"]["content"])

            if t["assistant"].get("sql"):
                with st.expander("🔎 실행된 SQL"):
                    st.code(t["assistant"]["sql"], language="sql")

            if t["assistant"].get("raw_sql"):
                with st.expander("🧪 원본 SQL"):
                    st.code(t["assistant"]["raw_sql"], language="sql")


# =====================================================
# 6.5) 시나리오 다음 작업(액션 칩) 표시
#   - out 변수를 여기서 쓰지 않음 (NameError 방지)
# =====================================================
clicked = render_action_chips(st.session_state.action_suggestions, key_prefix="next")
if clicked:
    st.session_state.pending_question = clicked
    st.session_state.action_suggestions = []
    st.rerun()


# =====================================================
# 7) 질문 입력
# =====================================================
user_input = st.chat_input("예: 이번 달 부서별 평균 근무시간은?")

question = None
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
elif user_input:
    question = user_input


# =====================================================
# 8) 실행 (Scenario → fallback SQL)
# =====================================================
if question:
    st.session_state.messages.append({"role": "user", "content": question})

    out = {}
    sql = None
    raw_sql = None

    try:
        spinner = show_center_spinner("처리 중...")

        out = orchestrator.run(
            session_id=st.session_state.session_id,
            user_text=question
        )

        spinner.empty()

        artifacts = out.get("artifacts") if isinstance(out.get("artifacts"), dict) else {}
        result = artifacts.get("result")

        is_scenario = out.get("state") is not None

        if is_scenario:
            # ✅ 시나리오 단계는 reply를 그대로(LLM 해설로 인한 오해 방지)
            answer = out.get("reply", "")

            # ✅ 다음 작업 가이드 저장 → 다음 rerun에서 버튼으로 표시
            st.session_state.action_suggestions = out.get("suggestions", []) or []
        else:
            # ✅ 일반 질의만 explainer 요약
            answer = explainer.invoke({"question": question, "result": result})

            # 일반 질의는 액션칩 비움
            st.session_state.action_suggestions = []

        sql = artifacts.get("fixed_sql")
        raw_sql = artifacts.get("raw_sql")

    except Exception as e:
        try:
            spinner.empty()
        except Exception:
            pass

        answer = f"❌ 오류: {e}"
        st.session_state.action_suggestions = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sql": sql,
        "raw_sql": raw_sql,
    })

    st.rerun()
