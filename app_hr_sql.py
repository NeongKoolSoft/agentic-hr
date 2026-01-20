# app_hr_sql.py
import uuid
import re
import ast
import os

import streamlit as st
import streamlit.components.v1 as components

from HR_sql_ai import HRTextToSQLEngine, ENGINE_VERSION
from scenario_payroll import ScenarioMemoryManager  # 메모리만 재사용

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from datetime import date, datetime

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError


# =====================================================
# DB 엔진(순수 SQLAlchemy) 유틸
# =====================================================
def _normalize_db_url(url: str) -> str:
    """
    SQLAlchemy는 'postgresql://'을 선호.
    (Supabase에서 'postgres://'로 주는 경우가 있어 보정)
    """
    if not url:
        return url
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


@st.cache_resource(show_spinner=False)
def get_db_engine() -> Engine:
    """
    ✅ 안전 패턴 핵심
    - st.cache_resource로 엔진 1회 생성/재사용
    - import 시점이 아니라 "처음 DB가 필요할 때" 호출되게 사용
    - pool_pre_ping로 죽은 커넥션 자동 감지
    - connect_timeout으로 무한 대기 방지
    - (권장) sslmode=require (Supabase는 보통 SSL 필요)
    """
    db_url = _normalize_db_url(os.getenv("DATABASE_URL", "").strip())
    if not db_url:
        raise RuntimeError("DATABASE_URL 환경변수가 설정되어 있지 않습니다.")

    connect_args = {"connect_timeout": 10}
    connect_args["sslmode"] = os.getenv("DB_SSLMODE", "require")

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),  # 30분
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
        connect_args=connect_args,
        future=True,
    )
    return engine


def db_ping(engine: Engine, retries: int = 3, backoff_sec: float = 1.2) -> None:
    """
    부팅 시/버튼 실행 시 'DB 연결 살아있나' 빠르게 체크하고 싶을 때.
    Render free/cold start에서 잠깐 안 붙는 경우가 있어 재시도 포함.
    """
    import time

    last_err = None
    for i in range(retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("select 1"))
            return
        except OperationalError as e:
            last_err = e
            time.sleep(backoff_sec * (i + 1))
    raise last_err


def fetch_all(sql: str, params: dict | None = None) -> list[dict]:
    """
    SELECT용 헬퍼: dict 리스트로 반환
    """
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.mappings().all()
    return [dict(r) for r in rows]


def execute(sql: str, params: dict | None = None) -> int:
    """
    INSERT/UPDATE/DELETE용 헬퍼: 영향 rowcount 반환
    """
    engine = get_db_engine()
    with engine.begin() as conn:
        result = conn.execute(text(sql), params or {})
    return int(result.rowcount or 0)


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


def request_scroll(target_id: str = "result-anchor"):
    st.session_state["_scroll_to_id"] = target_id


def run_scroll_if_requested():
    target_id = st.session_state.get("_scroll_to_id")
    if not target_id:
        return

    components.html(
        f"""
        <script>
          const doc = window.parent.document;
          const el = doc.getElementById("{target_id}");
          if (el) {{
            el.scrollIntoView({{ behavior: "smooth", block: "start" }});
          }}
        </script>
        """,
        height=0,
    )

    # 실행 후 제거
    del st.session_state["_scroll_to_id"]


def _month_bounds(d: date):
    """d가 속한 달의 [월초, 다음달월초) 반환"""
    month_start = d.replace(day=1)
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    return month_start, next_month


def enforce_month_range_sql(sql: str) -> str:
    """
    LLM이 pay_month = DATE 'YYYY-MM-DD' 같은 '일자 박기'를 만들면
    월 범위로 강제 변환:
      pay_month = DATE '2026-01-08'
      -> pay_month >= DATE '2026-01-01' AND pay_month < DATE '2026-02-01'
    """
    if not sql:
        return sql

    s = sql

    # 1) pay_month = DATE 'YYYY-MM-DD'
    pat1 = re.compile(
        r"(pay_month\s*=\s*DATE\s*'(\d{4}-\d{2}-\d{2})')",
        flags=re.IGNORECASE
    )

    def repl1(m):
        dt = datetime.strptime(m.group(2), "%Y-%m-%d").date()
        ms, nm = _month_bounds(dt)
        return f"pay_month >= DATE '{ms:%Y-%m-%d}' AND pay_month < DATE '{nm:%Y-%m-%d}'"

    s = pat1.sub(repl1, s)

    # 2) pay_month = 'YYYY-MM-DD'::date
    pat2 = re.compile(
        r"(pay_month\s*=\s*'(\d{4}-\d{2}-\d{2})'\s*::\s*date)",
        flags=re.IGNORECASE
    )

    def repl2(m):
        dt = datetime.strptime(m.group(2), "%Y-%m-%d").date()
        ms, nm = _month_bounds(dt)
        return f"pay_month >= DATE '{ms:%Y-%m-%d}' AND pay_month < DATE '{nm:%Y-%m-%d}'"

    s = pat2.sub(repl2, s)

    # 3) pay_month = DATE('YYYY-MM-DD')
    pat3 = re.compile(
        r"pay_month\s*=\s*DATE\s*\(\s*'(\d{4}-\d{2}-\d{2})'\s*\)",
        flags=re.IGNORECASE
    )

    def repl3(m):
        dt = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        ms, nm = _month_bounds(dt)
        return f"pay_month >= DATE '{ms:%Y-%m-%d}' AND pay_month < DATE '{nm:%Y-%m-%d}'"

    s = pat3.sub(repl3, s)

    return s


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
        padding-top: 0.55rem !important;
        padding-bottom: 1rem;
    }
    @media (max-width: 768px) {
        .block-container { padding-top: 0.35rem !important; }
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
        padding: 22px 28px;
        border-radius: 14px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        text-align: center;
        font-size: 15px;
        font-weight: 700;
        min-width: 240px;
    }
    </style>
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

# 시나리오 다음 작업(칩) 저장소
if "action_suggestions" not in st.session_state:
    st.session_state.action_suggestions = []

# RPC 시나리오 메모리 namespace
if "scenario_memory" not in st.session_state:
    st.session_state.scenario_memory = {}


def show_center_spinner(text: str = "처리 중..."):
    return st.markdown(
        f"""
        <div class="nk-overlay"></div>
        <div class="nk-center-spinner">⏳ {text}</div>
        """,
        unsafe_allow_html=True
    )


# ===============================
# 2) 환경변수 로드
# ===============================
def get_google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY")


def get_db_uri() -> str | None:
    return os.getenv("SUPABASE_DB_URI")

api_key = get_google_api_key()
db_uri = get_db_uri()

# ===============================
# 3) 환경변수 검증
# ===============================
if not api_key:
    st.error("❌ GOOGLE_API_KEY가 설정되어 있지 않습니다. (Render: Environment Variables 확인)")
    st.stop()

if not db_uri:
    st.error("❌ DATABASE_URL이 설정되어 있지 않습니다. (Render: Environment Variables 확인)")
    st.stop()

if "YOUR-PASSWORD" in db_uri:
    st.error("❌ DATABASE_URL에 [YOUR-PASSWORD]가 그대로 있습니다.")
    st.stop()


# =====================================================
# 3) HR/LLM 엔진 + Explainer
# =====================================================
@st.cache_resource(show_spinner=False)
def get_hr_engine(_db_uri: str, _api_key: str, _version: str) -> HRTextToSQLEngine:
    return HRTextToSQLEngine(db_uri=_db_uri, api_key=_api_key)


def ensure_hr_engine() -> HRTextToSQLEngine:
    """
    ✅ 전역 engine 제거 핵심:
    - 필요할 때만 가져오고
    - 캐시는 st.cache_resource가 처리
    """
    return get_hr_engine(db_uri, api_key, ENGINE_VERSION)

@st.cache_resource(show_spinner=False)
def get_explainer(_api_key: str):
    prompt = ChatPromptTemplate.from_template(
        """당신은 '넝쿨 HR 데이터 에이전트'입니다. 제공된 SQL 결과 데이터를 바탕으로 사용자에게 전문적이고 통찰력 있는 보고를 수행하세요.

        [답변 가이드라인]
        1. **결론 중심**: 데이터 조회 결과를 한 줄로 요약하며 시작하세요.
        2. **가독성**: 숫자나 리스트는 마크다운 표나 불렛 포인트를 활용해 한눈에 들어오게 하세요.
        3. **인사이트**: 데이터에서 읽을 수 있는 비즈니스적 의미(예: 전월 대비 변화, 특정 부서 집중 현상 등)를 짧게 언급하세요.
        4. **제언**: 분석 결과를 바탕으로 사용자가 다음에 확인해야 할 질문이나 액션을 제안하세요.
        5. **데이터 부재**: 결과가 없는 경우, 단순히 없다고 하기보다 '현재 조건으로는 데이터를 찾을 수 없으니, 기간이나 대상을 변경해보시는 것은 어떨까요?'와 같이 유연하게 대응하세요.

        질문: {question}
        SQL 결과: {result}

        데이터 에이전트의 답변:"""
    )
    return (
        prompt
        | ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=_api_key,
            temperature=0.2
        )
        | StrOutputParser()
    )


explainer = get_explainer(api_key)

# ... (기존 get_explainer 함수 아래에 추가) ...

@st.cache_resource(show_spinner=False)
def get_rewriter(_api_key: str):
    """
    [대화 맥락 유지 핵심]
    사용자의 불완전한 질문(예: "그럼 이건?")을 이전 대화 기록을 참고하여
    '완전한 문장'으로 다시 작성해주는 체인입니다.
    """
    prompt = ChatPromptTemplate.from_template(
        """당신은 사용자의 질문을 데이터베이스 조회를 위한 '완전한 질문'으로 재구성하는 AI입니다.
        
        [대화 기록]
        {history}
        
        [현재 질문]
        {question}
        
        위 대화 흐름을 고려하여, [현재 질문]을 SQL 생성이 가능한 '구체적이고 독립적인 질문'으로 다시 작성하세요.
        - 대명사("그것", "이전 것")가 있다면 명확한 명사로 바꾸세요.
        - 조건("반대로", "True만")이 변경되었다면 전체 문장에 반영하세요.
        - 질문의 의도가 바뀌지 않도록 주의하세요.
        - 설명 없이 오직 '재작성된 질문'만 출력하세요.
        
        재작성된 질문:"""
    )
    return (
        prompt
        | ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=_api_key, 
            temperature=0.1
        )
        | StrOutputParser()
    )

def format_history(messages, limit=6):
    """
    세션에 저장된 메시지 중 최근 N개를 텍스트로 변환합니다.
    """
    history_text = ""
    # 너무 오래된 기억은 버리고 최근 3턴(6개) 정도만 참조
    recent_msgs = messages[-limit:] if len(messages) > limit else messages
    
    for msg in recent_msgs:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        history_text += f"{role}: {content}\n"
    
    return history_text

# =====================================================
# 4) (RPC 전용) 결과 파서 / SQL 실행 유틸
# =====================================================
def _to_rows(result):
    """
    QuerySQLDatabaseTool 결과는 문자열로 오는 경우가 많음.
    예) "[(UUID('...'), 'DONE', {...})]" / "[(1, Decimal('123'))]"
    """
    if result is None:
        return []

    if isinstance(result, (list, tuple)):
        return list(result)

    if isinstance(result, str):
        s = result.strip()
        s = re.sub(r"Decimal\('(-?\d+(?:\.\d+)?)'\)", r"\1", s)
        s = re.sub(r'Decimal\("(-?\d+(?:\.\d+)?)"\)', r"\1", s)
        s = re.sub(r"UUID\('([0-9a-fA-F-]+)'\)", r"'\1'", s)
        s = s.replace("NULL", "None")

        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]
        except Exception:
            return []

    return []


def exec_sql(sql: str):
    """
    ✅ 전역 engine 제거:
    - 필요할 때 HR 엔진을 가져와 executor로 실행
    """
    hr = ensure_hr_engine()
    return hr.executor.invoke({"query": sql})


def fmt_won(n):
    try:
        return f"{int(float(n)):,}원"
    except Exception:
        return str(n)


# =====================================================
# 5) RPC 급여 시나리오: 슬롯 추출 (간단)
# =====================================================
TODAY_Y = 2026
TODAY_M = 1


def extract_period(text: str):
    t = text.strip()

    m = re.search(r"\b(20\d{2})[-./](0?[1-9]|1[0-2])\b", t)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"

    m = re.search(r"\b(20\d{2})\s*년\s*(0?[1-9]|1[0-2])\s*월\b", t)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"

    m = re.search(r"\b(0?[1-9]|1[0-2])\s*월\b", t)
    if m:
        return f"{TODAY_Y}-{int(m.group(1)):02d}"

    if re.search(r"(이번\s*달|당월|이번달)", t):
        return f"{TODAY_Y}-{TODAY_M:02d}"

    return None


def extract_scope(text: str):
    t = text.strip()

    if re.search(r"(전\s*직원|전체\s*직원|전체|전사|모두|전부서|전\s*부서|전부\s*서)", t):
        return "ALL"

    m = re.search(r"\b([가-힣A-Za-z0-9_]+)\s*(부|팀)\b", t)
    if m:
        return f"dept:{m.group(1)}{m.group(2)}"

    return None


def extract_date_any(text: str):
    """yyyy-mm-dd 또는 m/d 를 찾아 date string으로 반환(년은 period로 추론 가능)"""
    t = text.strip()

    m = re.search(r"\b(20\d{2})[-./](0?[1-9]|1[0-2])[-./](0?[1-9]|[12]\d|3[01])\b", t)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    m = re.search(r"\b(0?[1-9]|1[0-2])\s*/\s*(0?[1-9]|[12]\d|3[01])\b", t)
    if m:
        return f"__MD__:{int(m.group(1))}:{int(m.group(2))}"

    m = re.search(r"\b(0?[1-9]|[12]\d|3[01])\s*일\b", t)
    if m:
        return f"__DAY__:{int(m.group(1))}"

    return None


def extract_confirm(text: str):
    t = text.strip()
    if re.fullmatch(r"(예|네|응|진행|실행|확정|ok|ㅇㅋ)", t, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"(아니오|아니|취소|중단|no|ㄴㄴ)", t, flags=re.IGNORECASE):
        return False
    return None


def is_rpc_trigger(text: str):
    return bool(re.search(r"(급여|세금|공제|지급|이체|송금|전표|분개)", text)) and (
        is_execute_intent(text) or not is_query_intent(text)
    )


def is_execute_intent(text: str) -> bool:
    t = text.strip()
    return bool(re.search(r"(처리|실행|진행|계산|산정해|돌려|생성해|등록|전표생성|지급해)", t))


def is_query_intent(text: str) -> bool:
    t = text.strip()
    return bool(re.search(r"(몇\s*명|인원|대상|총액|합계|금액|건수|결과|내역|리스트|상세|조회|보여줘)", t))


def month_to_period_date(period_yyyy_mm: str):
    y, m = period_yyyy_mm.split("-")
    return f"{int(y):04d}-{int(m):02d}-01"


# =====================================================
# 6) RPC 시나리오 오케스트레이터(최소)
# =====================================================
RPC_ACTIVE = "PAYROLL_RPC"
S_PAYROLL = "PAYROLL"
S_TAX = "TAX"
S_PAYMENT = "PAYMENT"
S_JOURNAL = "JOURNAL"
S_DONE = "DONE"

STATE_LABEL = {
    S_PAYROLL: "급여 산정(RPC)",
    S_TAX: "공제 검증(RPC)",
    S_PAYMENT: "지급 처리(RPC)",
    S_JOURNAL: "전표 생성(RPC)",
    S_DONE: "완료(RPC)",
}

memory = ScenarioMemoryManager(store=st.session_state, namespace="scenario_memory")


def rpc_get_ctx(session_id: str) -> dict:
    return memory.get(session_id) or {}


def rpc_set_ctx(session_id: str, ctx: dict):
    memory.set(session_id, ctx)


def rpc_clear_ctx(session_id: str):
    memory.clear(session_id)


def rpc_fetch_run(run_id: str):
    sql = f"""
    select run_id, process_type, period, scope, status, params, summary, error_msg, started_at, finished_at
    from public.process_runs
    where run_id = '{run_id}';
    """
    return exec_sql(sql), sql.strip()


def rpc_fetch_lines(run_id: str):
    sql = f"""
    select line_id, line_type, data, created_at
    from public.process_run_lines
    where run_id = '{run_id}'
    order by line_id;
    """
    return exec_sql(sql), sql.strip()


def rpc_answer_query_from_refs(ctx: dict, user_text: str):
    refs = (ctx or {}).get("refs", {}) or {}

    ask_headcount = bool(re.search(r"(인원|몇\s*명|대상)", user_text))
    ask_total_gross = bool(re.search(r"(총\s*급여|총급여|gross)", user_text))
    ask_total_net = bool(re.search(r"(총\s*실지급|실지급|net)", user_text))
    ask_total_ded = bool(re.search(r"(총\s*공제|공제\s*총액|deduction)", user_text))
    ask_payment_lines = bool(re.search(r"(지급\s*라인|지급\s*내역|지급\s*건수|이체\s*건수)", user_text))
    ask_journal_lines = bool(re.search(r"(전표\s*라인|전표\s*내역|분개\s*내역|전표\s*건수)", user_text))

    payroll_run_id = refs.get("payroll_run_id")
    tax_run_id = refs.get("tax_run_id")
    payment_run_id = refs.get("payment_run_id")
    journal_run_id = refs.get("journal_run_id")

    target_run_id = payroll_run_id
    if re.search(r"(공제|세금)", user_text) and tax_run_id:
        target_run_id = tax_run_id
    if re.search(r"(지급|이체|송금)", user_text) and payment_run_id:
        target_run_id = payment_run_id
    if re.search(r"(전표|분개)", user_text) and journal_run_id:
        target_run_id = journal_run_id

    if not target_run_id:
        return None

    run_row_res, sql_fetch = rpc_fetch_run(str(target_run_id))
    rr = _to_rows(run_row_res)
    summary = {}
    if rr and isinstance(rr[0], (list, tuple)) and len(rr[0]) >= 7:
        summary = rr[0][6] if isinstance(rr[0][6], dict) else {}

    if ask_headcount:
        base_id = payroll_run_id or target_run_id
        base_res, base_sql = rpc_fetch_run(str(base_id))
        br = _to_rows(base_res)
        base_summary = {}
        if br and isinstance(br[0], (list, tuple)) and len(br[0]) >= 7:
            base_summary = br[0][6] if isinstance(br[0][6], dict) else {}
        n = base_summary.get("employee_count")
        reply = f"📌 급여 산정 대상 인원: **{n}명**"
        return {"reply": reply, "sqls": [base_sql]}

    if ask_total_gross:
        v = summary.get("total_gross")
        return {"reply": f"📌 총급여: **{fmt_won(v)}**", "sqls": [sql_fetch]}

    if ask_total_ded:
        v = summary.get("total_deductions")
        return {"reply": f"📌 총공제: **{fmt_won(v)}**", "sqls": [sql_fetch]}

    if ask_total_net:
        v = summary.get("total_net_pay") or summary.get("pay_total")
        return {"reply": f"📌 총실지급: **{fmt_won(v)}**", "sqls": [sql_fetch]}

    if ask_payment_lines and payment_run_id:
        lines_res, sql_lines = rpc_fetch_lines(str(payment_run_id))
        rows = _to_rows(lines_res)
        cnt = len(rows)
        return {"reply": f"📌 지급 라인 건수: **{cnt}건**", "sqls": [sql_lines]}

    if ask_journal_lines and journal_run_id:
        lines_res, sql_lines = rpc_fetch_lines(str(journal_run_id))
        rows = _to_rows(lines_res)
        cnt = len(rows)
        return {"reply": f"📌 전표 라인 건수: **{cnt}건**", "sqls": [sql_lines]}

    return {"reply": f"📌 요약: {summary}", "sqls": [sql_fetch]}


def rpc_run(session_id: str, user_text: str) -> dict:
    ctx = rpc_get_ctx(session_id)
    active = ctx.get("active_scenario") == RPC_ACTIVE
    confirm = extract_confirm(user_text)

    if re.search(r"(취소|종료|그만|중단|리셋|초기화)", user_text):
        rpc_clear_ctx(session_id)
        return {"handled": True, "reply": "RPC 급여 시나리오를 종료했습니다.", "state": None,
                "suggestions": [], "artifacts": {"rpc_sqls": []}}

    if not active:
        ctx = {
            "active_scenario": RPC_ACTIVE,
            "state": S_PAYROLL,
            "slots": {},
            "refs": {},
            "history": [],
        }

    if is_query_intent(user_text) and confirm is None and ctx.get("refs"):
        q = rpc_answer_query_from_refs(ctx, user_text)
        if q:
            rpc_set_ctx(session_id, ctx)
            return {"handled": True, "reply": q["reply"], "state": ctx.get("state"),
                    "suggestions": ["전체 프로세스 요약", "시나리오 종료"],
                    "artifacts": {"rpc_sqls": q.get("sqls", [])}}

    slots = ctx.get("slots", {})

    if is_query_intent(user_text) and not is_execute_intent(user_text) and confirm is None:
        if ctx.get("refs"):
            q = rpc_answer_query_from_refs(ctx, user_text)
            if q:
                rpc_set_ctx(session_id, ctx)
                return {
                    "handled": True,
                    "reply": q["reply"],
                    "state": ctx.get("state"),
                    "suggestions": ["전체 프로세스 요약", "시나리오 종료"],
                    "artifacts": {"rpc_sqls": q.get("sqls", [])},
                }

    period = extract_period(user_text)
    scope = extract_scope(user_text)
    any_date = extract_date_any(user_text)

    if period:
        slots["period"] = period
    if scope:
        slots["scope"] = scope

    if any_date:
        if re.search(r"(전표|분개|전기)", user_text):
            slots["journal_date_raw"] = any_date
        elif re.search(r"(지급|이체|송금)", user_text):
            slots["pay_date_raw"] = any_date
        else:
            slots["pay_date_raw"] = any_date

    ctx["slots"] = slots

    def resolve_md(raw, period_yyyy_mm):
        if not raw:
            return None
        if raw.startswith("__MD__:"):
            _, mm, dd = raw.split(":")
            y = int(period_yyyy_mm.split("-")[0])
            return f"{y:04d}-{int(mm):02d}-{int(dd):02d}"
        if raw.startswith("__DAY__:"):
            dd = int(raw.split(":")[1])
            y, m = period_yyyy_mm.split("-")
            return f"{int(y):04d}-{int(m):02d}-{dd:02d}"
        return raw

    state = ctx.get("state") or S_PAYROLL
    rpc_sqls = []

    period_yyyy_mm = slots.get("period")
    scope_val = slots.get("scope")

    # -------------------------
    # S_PAYROLL
    # -------------------------
    if state == S_PAYROLL:
        if not period_yyyy_mm or not scope_val:
            miss = []
            if not period_yyyy_mm: miss.append("period(예: 2026년 1월)")
            if not scope_val: miss.append("scope(예: 전직원/영업부)")
            reply = (
                "RPC 급여(프로시저) 실행을 위해 정보가 필요합니다.\n"
                f"- 누락: {', '.join(miss)}\n"
                "- 예: '2026년 1월 전직원 급여 처리'\n"
                "- 예: '1월 영업부 급여 처리'"
            )
            ctx["state"] = S_PAYROLL
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": reply,
                "state": ctx["state"],
                "suggestions": ["2026년 1월 전직원 급여 처리", "이번달 전직원 급여 처리", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        period_date = month_to_period_date(period_yyyy_mm)
        sql_call = f"select public.rpc_payroll_run('{period_date}'::date, '{scope_val}') as run_id;"
        run_id_res = exec_sql(sql_call)
        rpc_sqls.append(sql_call)

        rows = _to_rows(run_id_res)
        run_id = None
        if rows and isinstance(rows[0], (list, tuple)) and len(rows[0]) >= 1:
            run_id = rows[0][0]
        elif rows and isinstance(rows[0], str):
            run_id = rows[0]

        if not run_id:
            ctx["state"] = S_PAYROLL
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "급여 RPC 호출은 실행했지만 run_id를 파싱하지 못했습니다. (DB 반환값 확인 필요)",
                "state": ctx["state"],
                "suggestions": ["다시 시도", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls, "result": run_id_res},
            }

        ctx["refs"]["payroll_run_id"] = str(run_id)
        ctx["history"].append({"state": S_PAYROLL, "run_id": str(run_id)})

        run_row_res, sql_fetch = rpc_fetch_run(str(run_id))
        rpc_sqls.append(sql_fetch)

        rr = _to_rows(run_row_res)
        summary = {}
        status = None
        if rr and isinstance(rr[0], (list, tuple)) and len(rr[0]) >= 7:
            status = rr[0][4]
            summary = rr[0][6] if isinstance(rr[0][6], dict) else {}

        ctx["state"] = S_TAX
        rpc_set_ctx(session_id, ctx)

        reply = (
            "✅ [RPC] 급여 산정 실행 완료\n"
            f"- run_id: {run_id}\n"
        )
        if summary:
            reply += (
                f"- 대상 인원: {summary.get('employee_count')}명\n"
                f"- 총급여: {fmt_won(summary.get('total_gross'))}\n"
                f"- 총공제: {fmt_won(summary.get('total_deductions'))}\n"
                f"- 총실지급: {fmt_won(summary.get('total_net_pay'))}\n"
            )
        reply += "\n다음 단계로 **공제 검증(RPC)** 을 진행할까요?"

        return {
            "handled": True,
            "reply": reply,
            "state": ctx["state"],
            "suggestions": ["공제 검증 진행", "시나리오 종료"],
            "artifacts": {"rpc_sqls": rpc_sqls, "run_id": str(run_id), "summary": summary, "status": status},
        }

    # -------------------------
    # S_TAX
    # -------------------------
    if state == S_TAX:
        if not period_yyyy_mm or not scope_val:
            ctx["state"] = S_PAYROLL
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "공제 검증 전에 period/scope가 필요합니다. 예: '2026년 1월 전직원 급여 처리'",
                "state": ctx["state"],
                "suggestions": ["2026년 1월 전직원 급여 처리", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        payroll_run_id = ctx["refs"].get("payroll_run_id")
        if not payroll_run_id:
            ctx["state"] = S_PAYROLL
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "공제 검증 전에 급여 실행(run_id)이 필요합니다. 먼저 '급여 처리'부터 해줘.",
                "state": ctx["state"],
                "suggestions": ["급여 처리", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        period_date = month_to_period_date(period_yyyy_mm)
        sql_call = f"select public.rpc_tax_run('{period_date}'::date, '{scope_val}', '{payroll_run_id}'::uuid) as run_id;"
        run_id_res = exec_sql(sql_call)
        rpc_sqls.append(sql_call)

        rows = _to_rows(run_id_res)
        run_id = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else None

        ctx["refs"]["tax_run_id"] = str(run_id)
        ctx["history"].append({"state": S_TAX, "run_id": str(run_id)})

        run_row_res, sql_fetch = rpc_fetch_run(str(run_id))
        rpc_sqls.append(sql_fetch)

        rr = _to_rows(run_row_res)
        summary = {}
        if rr and isinstance(rr[0], (list, tuple)) and len(rr[0]) >= 7:
            summary = rr[0][6] if isinstance(rr[0][6], dict) else {}

        ctx["state"] = S_PAYMENT
        rpc_set_ctx(session_id, ctx)

        reply = (
            "✅ [RPC] 공제 검증 완료\n"
            f"- run_id: {run_id}\n"
        )
        if summary:
            rate = summary.get("avg_deduction_rate", 0)
            try:
                rate_pct = float(rate) * 100.0
            except Exception:
                rate_pct = rate
            reply += (
                f"- 총급여: {fmt_won(summary.get('total_gross'))}\n"
                f"- 총공제: {fmt_won(summary.get('total_deductions'))}\n"
                f"- 총실지급: {fmt_won(summary.get('total_net_pay'))}\n"
                f"- 평균 공제율: {rate_pct:.2f}%\n"
                f"- 공제 0원 인원: {summary.get('zero_deduction_count')}명\n"
            )
        reply += "\n다음 단계로 **지급 처리(RPC)** 를 진행할까요? 지급일을 입력해줘."

        return {
            "handled": True,
            "reply": reply,
            "state": ctx["state"],
            "suggestions": ["25일 지급", "2026-01-25 지급", "시나리오 종료"],
            "artifacts": {"rpc_sqls": rpc_sqls, "run_id": str(run_id), "summary": summary},
        }

    # -------------------------
    # S_PAYMENT
    # -------------------------
    if state == S_PAYMENT:
        tax_run_id = ctx["refs"].get("tax_run_id")
        if not tax_run_id:
            ctx["state"] = S_TAX
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "지급 처리 전에 공제 검증(run_id)이 필요합니다. '공제 검증 진행'을 먼저 해줘.",
                "state": ctx["state"],
                "suggestions": ["공제 검증 진행", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        if not period_yyyy_mm or not scope_val:
            ctx["state"] = S_PAYROLL
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "지급 처리 전에 period/scope가 필요합니다. '2026년 1월 전직원 급여 처리'부터 진행해줘.",
                "state": ctx["state"],
                "suggestions": ["2026년 1월 전직원 급여 처리", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        pay_date = resolve_md(slots.get("pay_date_raw"), period_yyyy_mm)
        if not pay_date:
            ctx["state"] = S_PAYMENT
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "지급일이 필요합니다. 예: '25일 지급' 또는 '2026-01-25 지급'",
                "state": ctx["state"],
                "suggestions": ["25일 지급", "2026-01-25 지급", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        if confirm is None:
            ctx["state"] = S_PAYMENT
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": (
                    "지급 실행(배치 생성)을 진행할까요?\n"
                    f"- period={period_yyyy_mm}\n"
                    f"- scope={scope_val}\n"
                    f"- pay_date={pay_date}\n\n"
                    "예/아니오"
                ),
                "state": ctx["state"],
                "suggestions": ["예", "아니오", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        if confirm is False:
            ctx["state"] = S_PAYMENT
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "지급 실행을 취소했습니다. (계속하려면 '예' 또는 지급일을 다시 입력해줘)",
                "state": ctx["state"],
                "suggestions": ["예", "25일 지급", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        period_date = month_to_period_date(period_yyyy_mm)
        sql_call = (
            f"select public.rpc_payment_run('{period_date}'::date, '{scope_val}', "
            f"'{tax_run_id}'::uuid, '{pay_date}'::date) as run_id;"
        )
        run_id_res = exec_sql(sql_call)
        rpc_sqls.append(sql_call)

        rows = _to_rows(run_id_res)
        run_id = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else None

        ctx["refs"]["payment_run_id"] = str(run_id)
        ctx["history"].append({"state": S_PAYMENT, "run_id": str(run_id)})

        run_row_res, sql_fetch = rpc_fetch_run(str(run_id))
        rpc_sqls.append(sql_fetch)

        rr = _to_rows(run_row_res)
        summary = {}
        if rr and isinstance(rr[0], (list, tuple)) and len(rr[0]) >= 7:
            summary = rr[0][6] if isinstance(rr[0][6], dict) else {}

        ctx["state"] = S_JOURNAL
        rpc_set_ctx(session_id, ctx)

        reply = (
            "✅ [RPC] 지급 처리 완료\n"
            f"- run_id: {run_id}\n"
        )
        if summary:
            reply += (
                f"- 성공 대상: {summary.get('success_count')}명\n"
                f"- 오류: {summary.get('error_count')}건\n"
                f"- 지급총액: {fmt_won(summary.get('pay_total'))}\n"
                f"- 지급일: {summary.get('pay_date')}\n"
            )
        reply += "\n다음 단계로 **전표 생성(RPC)** 을 진행할까요? 전표일을 입력해줘."

        return {
            "handled": True,
            "reply": reply,
            "state": ctx["state"],
            "suggestions": ["2026-01-31 전표", "1/31 전표", "시나리오 종료"],
            "artifacts": {"rpc_sqls": rpc_sqls, "run_id": str(run_id), "summary": summary},
        }

    # -------------------------
    # S_JOURNAL
    # -------------------------
    if state == S_JOURNAL:
        payment_run_id = ctx["refs"].get("payment_run_id")
        if not payment_run_id:
            ctx["state"] = S_PAYMENT
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "전표 생성 전에 지급 처리(run_id)가 필요합니다. 먼저 '지급'부터 진행해줘.",
                "state": ctx["state"],
                "suggestions": ["25일 지급", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        if not period_yyyy_mm or not scope_val:
            ctx["state"] = S_PAYROLL
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "전표 생성 전에 period/scope가 필요합니다. '2026년 1월 전직원 급여 처리'부터 진행해줘.",
                "state": ctx["state"],
                "suggestions": ["2026년 1월 전직원 급여 처리", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        journal_date = resolve_md(slots.get("journal_date_raw"), period_yyyy_mm)
        if not journal_date:
            ctx["state"] = S_JOURNAL
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "전표일이 필요합니다. 예: '2026-01-31 전표' 또는 '1/31 전표'",
                "state": ctx["state"],
                "suggestions": ["2026-01-31 전표", "1/31 전표", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        if confirm is None:
            ctx["state"] = S_JOURNAL
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": (
                    "전표 생성을 진행할까요? (전표 초안 생성)\n"
                    f"- period={period_yyyy_mm}\n"
                    f"- scope={scope_val}\n"
                    f"- journal_date={journal_date}\n\n"
                    "예/아니오"
                ),
                "state": ctx["state"],
                "suggestions": ["예", "아니오", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        if confirm is False:
            ctx["state"] = S_JOURNAL
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "전표 생성을 취소했습니다. (계속하려면 '예' 또는 전표일을 다시 입력해줘)",
                "state": ctx["state"],
                "suggestions": ["예", "2026-01-31 전표", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        period_date = month_to_period_date(period_yyyy_mm)
        sql_call = (
            f"select public.rpc_journal_post('{period_date}'::date, '{scope_val}', "
            f"'{payment_run_id}'::uuid, '{journal_date}'::date) as run_id;"
        )
        run_id_res = exec_sql(sql_call)
        rpc_sqls.append(sql_call)

        rows = _to_rows(run_id_res)
        run_id = rows[0][0] if rows and isinstance(rows[0], (list, tuple)) else None

        ctx["refs"]["journal_run_id"] = str(run_id)
        ctx["history"].append({"state": S_JOURNAL, "run_id": str(run_id)})

        run_row_res, sql_fetch = rpc_fetch_run(str(run_id))
        rpc_sqls.append(sql_fetch)

        rr = _to_rows(run_row_res)
        summary = {}
        if rr and isinstance(rr[0], (list, tuple)) and len(rr[0]) >= 7:
            summary = rr[0][6] if isinstance(rr[0][6], dict) else {}

        lines_res, sql_lines = rpc_fetch_lines(str(run_id))
        rpc_sqls.append(sql_lines)

        ctx["state"] = S_DONE
        rpc_set_ctx(session_id, ctx)

        reply = (
            "✅ [RPC] 전표 생성 완료(초안)\n"
            f"- run_id: {run_id}\n"
        )
        if summary:
            reply += (
                f"- 차변 합계: {fmt_won(summary.get('debit_total'))}\n"
                f"- 대변 합계: {fmt_won(summary.get('credit_total'))}\n"
                f"- 차대일치: {summary.get('balanced')}\n"
                f"- 전표일: {summary.get('journal_date')}\n"
            )
        reply += "\n전체 프로세스 요약을 보여드릴까요? (예/아니오)"

        return {
            "handled": True,
            "reply": reply,
            "state": ctx["state"],
            "suggestions": ["예", "아니오", "시나리오 종료"],
            "artifacts": {
                "rpc_sqls": rpc_sqls,
                "run_id": str(run_id),
                "summary": summary,
                "lines_result": lines_res,
            },
        }

    # -------------------------
    # S_DONE
    # -------------------------
    if state == S_DONE:
        if is_query_intent(user_text) and not is_execute_intent(user_text) and confirm is None:
            if ctx.get("refs"):
                q = rpc_answer_query_from_refs(ctx, user_text)
                if q:
                    rpc_set_ctx(session_id, ctx)
                    return {
                        "handled": True,
                        "reply": q["reply"],
                        "state": ctx.get("state"),
                        "suggestions": ["전체 프로세스 요약", "시나리오 종료"],
                        "artifacts": {"rpc_sqls": q.get("sqls", [])},
                    }

        if re.search(r"(전체\s*요약|요약\s*보여줘|요약)", user_text) and confirm is None:
            confirm = True

        if confirm is None:
            ctx["state"] = S_DONE
            rpc_set_ctx(session_id, ctx)
            return {
                "handled": True,
                "reply": "전체 프로세스 요약을 보여드릴까요? (예/아니오)",
                "state": ctx["state"],
                "suggestions": ["예", "아니오", "시나리오 종료"],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        if confirm is False:
            rpc_clear_ctx(session_id)
            return {
                "handled": True,
                "reply": "알겠습니다. RPC 시나리오를 종료했습니다.",
                "state": None,
                "suggestions": [],
                "artifacts": {"rpc_sqls": rpc_sqls},
            }

        refs = ctx.get("refs", {})
        reply = (
            "✅ [RPC] 급여 → 공제 → 지급 → 전표 요약\n"
            f"- payroll_run_id: {refs.get('payroll_run_id')}\n"
            f"- tax_run_id: {refs.get('tax_run_id')}\n"
            f"- payment_run_id: {refs.get('payment_run_id')}\n"
            f"- journal_run_id: {refs.get('journal_run_id')}\n"
        )
        rpc_clear_ctx(session_id)
        return {
            "handled": True,
            "reply": reply,
            "state": None,
            "suggestions": [],
            "artifacts": {"rpc_sqls": rpc_sqls},
        }

    ctx["state"] = S_PAYROLL
    rpc_set_ctx(session_id, ctx)
    return {
        "handled": True,
        "reply": "상태가 꼬여서 처음 단계로 돌아갑니다. '2026년 1월 전직원 급여 처리'로 시작해줘.",
        "state": S_PAYROLL,
        "suggestions": ["2026년 1월 전직원 급여 처리"],
        "artifacts": {"rpc_sqls": rpc_sqls},
    }


# =====================================================
# 7) 헤더
# =====================================================
st.markdown(
    """
    <div style="text-align:center; padding:15px 0 2px 0;">
      <h2 style="margin:0;">Agentic AI for 넝쿨HR</h2>
      <div style="font-size:12px; opacity:0.75; margin-top:2px;">
        조회는 LLM(SQL 생성) · 급여 프로세스는 Supabase RPC 실행
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 🔀 조회 / RPC 실행 모드 선택
# =====================================================
with st.container():
    st.checkbox(
        "🔁 급여·공제·지급·전표 **실행 모드** (체크 해제 시 결과 조회)",
        key="rpc_execute_mode",
        value=False
    )

# =====================================================
# 8) 🧭 RPC 시나리오 상태 바 + 종료
# =====================================================
ctx_rpc = rpc_get_ctx(st.session_state.session_id)
if ctx_rpc and ctx_rpc.get("active_scenario") == RPC_ACTIVE:
    state = ctx_rpc.get("state")
    st.info(f"🧭 현재 작업: 급여 처리(RPC) · 단계: {STATE_LABEL.get(state, state)}")

    if st.button("❌ 시나리오 종료", key="rpc_exit"):
        rpc_clear_ctx(st.session_state.session_id)
        st.session_state.action_suggestions = []
        st.success("시나리오가 종료되었습니다.")
        st.rerun()

# =====================================================
# 9) 대표 질문
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
    "2026년 1월 전직원 급여 처리해줘",
]

cols = st.columns(2)
for i, q in enumerate(chip_questions):
    with cols[i % 2]:
        if st.button(q, use_container_width=True, key=f"chip_{i}"):
            st.session_state.pending_question = q
            st.rerun()

st.divider()

# =====================================================
# 10) 기존 대화 표시
# =====================================================
turns = build_turns(st.session_state.messages)

last_sql_turn_idx = -1
for i, t in enumerate(turns):
    a = t.get("assistant") or {}
    if a.get("sql") or a.get("raw_sql"):
        last_sql_turn_idx = i

for i, t in enumerate(turns):
    if t["user"]:
        with st.chat_message("user"):
            st.markdown(t["user"]["content"])

    if t["assistant"]:
        with st.chat_message("assistant"):
            st.markdown(t["assistant"]["content"])
            expand_this = (i == last_sql_turn_idx)

            if t["assistant"].get("sql"):
                with st.expander("🔎 실행된 SQL", expanded=expand_this):
                    st.code(t["assistant"]["sql"], language="sql")

st.markdown('<div id="result-anchor"></div>', unsafe_allow_html=True)
run_scroll_if_requested()

# =====================================================
# 10.5) 시나리오 다음 작업(액션 칩) 표시
# =====================================================
clicked = render_action_chips(st.session_state.action_suggestions, key_prefix="next")
if clicked:
    st.session_state.pending_question = clicked
    st.session_state.action_suggestions = []
    st.rerun()

# =====================================================
# 11) 질문 입력
# =====================================================
user_input = st.chat_input("예: 이번 달 부서별 평균 근무시간은? / 2026년 1월 전직원 급여 처리")

question = None
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
elif user_input:
    question = user_input

# =====================================================
# 12) 실행: (RPC 시나리오 우선) → fallback LLM 조회
# =====================================================
if question:
    st.session_state.messages.append({"role": "user", "content": question})

    answer = ""
    sql_to_show = None
    raw_sql_to_show = None

    try:
        spinner = show_center_spinner("처리 중...")

        execute_mode = st.session_state.get("rpc_execute_mode", False)

        # 1) 실행 모드: RPC
        if execute_mode:
            out_rpc = rpc_run(st.session_state.session_id, question)

            if out_rpc.get("handled"):
                spinner.empty()
                answer = out_rpc.get("reply", "")
                st.session_state.action_suggestions = out_rpc.get("suggestions", []) or []

                rpc_sqls = (out_rpc.get("artifacts", {}) or {}).get("rpc_sqls", []) or []
                if rpc_sqls:
                    sql_to_show = "\n\n".join(s.strip() for s in rpc_sqls)

            else:
                spinner.empty()
                answer = "⚠️ 실행 모드입니다. 실행 가능한 명령을 입력해 주세요."
                st.session_state.action_suggestions = ["시나리오 종료"]

        # 2) 조회 모드: LLM SQL 조회
        else:
            hr = ensure_hr_engine()  # ✅ 전역 engine 대신 여기서 가져옴
            
            # [Step 1] 질문 재작성 (기억력 주입) 🧠
            # 대화 기록이 있을 때만 동작합니다.
            real_question = question
            if len(st.session_state.messages) > 0:
                rewriter = get_rewriter(api_key)
                history_str = format_history(st.session_state.messages[:-1]) # 방금 넣은 질문 제외
                
                # "아니, True만 보여줘" -> "야근 여부가 True인 사람만 보여줘" 로 변환
                real_question = rewriter.invoke({
                    "history": history_str, 
                    "question": question
                })
                print(f"🔄 Original: {question} -> Rewritten: {real_question}") # 디버깅용 로그

            # [Step 2] 변환된 질문(real_question)으로 SQL 생성
            out = hr.run(real_question)
            spinner.empty()

            fixed_sql = out.get("fixed_sql") or ""
            raw_sql = out.get("raw_sql")

            patched_sql = enforce_month_range_sql(fixed_sql)

            # ✅ 보정된 SQL로 직접 실행
            patched_result = exec_sql(patched_sql)

            # [Step 3] 결과 설명 (사용자에게는 원래 질문에 대한 답인 것처럼)
            answer = explainer.invoke({
                "question": real_question, # 설명할 때도 구체적인 질문을 줍니다.
                "result": patched_result
            })

            sql_to_show = patched_sql
            raw_sql_to_show = fixed_sql if raw_sql is None else raw_sql
            st.session_state.action_suggestions = []

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
        "sql": sql_to_show,
        "raw_sql": raw_sql_to_show,
    })

    request_scroll("result-anchor")
    st.rerun()
