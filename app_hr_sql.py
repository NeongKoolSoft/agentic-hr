# app_hr_sql.py
import uuid
import re
import ast
import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
import time
import base64
import fitz  # PyMuPDF
import hashlib
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
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
# 🔧 LLM 전역 설정 (모델 교체 대비)
# =====================================================

LLM_MODELS = {
    "FAST": {
        "model": "gemini-2.0-flash",
        "temperature": 0.0,
    },
    "REASONING": {
        "model": "gemini-2.0-flash",  # ← 3월 이후 여기만 바꾸면 됨
        "temperature": 0.2,
    }
}

LLM_TIMEOUT = 30
LLM_MAX_RETRIES = 2


# =====================================================
# DB 엔진(순수 SQLAlchemy) 유틸
# =====================================================
def _normalize_db_url(url: str) -> str:
    """
    SQLAlchemy에서 사용하는 데이터베이스 URL이 postgresql:// 포맷이어야 하므로,
    postgres://로 주어진 경우 자동으로 보정해준다.
    """
    if not url:
        return url
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


@st.cache_resource(show_spinner=False)
def get_db_engine() -> Engine:
    """
    환경변수에서 DB 접속 정보를 읽어 SQLAlchemy 엔진을 한 번만 생성하고 캐시한다.
    커넥션 풀/SSL 등 DB 연결안전설정을 적용해서 엔진 생성.
    """
    db_url = _normalize_db_url(os.getenv("SUPABASE_DB_URI", "").strip())
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URI 환경변수가 설정되어 있지 않습니다.")

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
    DB 연결이 살아있는지 빠르게 체크하는 유틸.
    엔진 커넥션이 임시로 죽었을 때 재시도(backoff 포함).
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
    SELECT 쿼리를 실행하여 dict형 리스트로 결과를 반환하는 헬퍼 함수.
    """
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.mappings().all()
    return [dict(r) for r in rows]


def execute(sql: str, params: dict | None = None) -> int:
    """
    INSERT/UPDATE/DELETE 쿼리를 실행 후 영향받은 row 개수를 반환하는 헬퍼 함수.
    """
    engine = get_db_engine()
    with engine.begin() as conn:
        result = conn.execute(text(sql), params or {})
    return int(result.rowcount or 0)


# =====================================================
# 유틸: 메시지 → 턴 구조
# =====================================================
def build_turns(messages):
    """
    메시지 배열을 user/assistant 기준의 턴 묶음 구조로 변환한다.
    즉, user→assistant 쌍을 하나의 turn으로 반환한다.
    """
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
    """
    지정된 id로 스크롤 이동을 트리거하는 플래그를 세션에 지정한다.
    """
    st.session_state["_scroll_to_id"] = target_id


def run_scroll_if_requested():
    """
    스크롤 요청 플래그가 있는 경우, HTML/js를 통해 해당 위치로 부드럽게 이동시킨다.
    """
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

    # 실행 후 플래그 삭제
    del st.session_state["_scroll_to_id"]


def _month_bounds(d: date):
    """
    일자가 속한 달의 월초, 다음달 월초(date)를 튜플로 반환한다.
    예: 2026-01-15 -> (2026-01-01, 2026-02-01)
    """
    month_start = d.replace(day=1)
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    return month_start, next_month


def enforce_month_range_sql(sql: str) -> str:
    """
    SQL 내부에 pay_month = 'YYYY-MM-DD' 처럼 '일자 박기' 조건이 있으면,
    pay_month가 속한 월 전체 범위로 치환(월초 ~ 다음달월초 미만)하여 반환한다.
    """
    if not sql:
        return sql

    s = sql

    # 1) pay_month = DATE 'YYYY-MM-DD' 패턴 치환
    pat1 = re.compile(
        r"(pay_month\s*=\s*DATE\s*'(\d{4}-\d{2}-\d{2})')",
        flags=re.IGNORECASE
    )

    def repl1(m):
        dt = datetime.strptime(m.group(2), "%Y-%m-%d").date()
        ms, nm = _month_bounds(dt)
        return f"pay_month >= DATE '{ms:%Y-%m-%d}' AND pay_month < DATE '{nm:%Y-%m-%d}'"

    s = pat1.sub(repl1, s)

    # 2) pay_month = 'YYYY-MM-DD'::date 패턴 치환
    pat2 = re.compile(
        r"(pay_month\s*=\s*'(\d{4}-\d{2}-\d{2})'\s*::\s*date)",
        flags=re.IGNORECASE
    )

    def repl2(m):
        dt = datetime.strptime(m.group(2), "%Y-%m-%d").date()
        ms, nm = _month_bounds(dt)
        return f"pay_month >= DATE '{ms:%Y-%m-%d}' AND pay_month < DATE '{nm:%Y-%m-%d}'"

    s = pat2.sub(repl2, s)

    # 3) pay_month = DATE('YYYY-MM-DD') 패턴 치환
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
    """
    시나리오가 제안하는 다음 행동(예/아니오/지급 진행 등)을 버튼 칩으로 화면에 표시하고
    클릭 시 해당 값을 리턴한다.
    """
    if not suggestions:
        return None

    shown = suggestions[:4]
    cols = st.columns(len(shown))
    for i, label in enumerate(shown):
        if cols[i].button(label, key=f"{key_prefix}_{i}_{label}", use_container_width=True):
            return label
    return None

def is_employment_cert_trigger(text: str) -> bool:
    """
    text 내용이 재직증명서 관련 요청인지 감지하는 함수.
    """
    t = (text or "").strip()
    return bool(re.search(r"(재직\s*증명서|재직증명서|증명서\s*출력|employment\s*certificate)", t, re.IGNORECASE))

def extract_employee_hint(text: str) -> str | None:
    """
    사용자가 '김철수 재직증명서' 등으로 입력 시 이름 추정 힌트만 뽑아주는 함수.
    재직증명서/출력/발급 등 키워드는 제거하여 남은 텍스트만 반환.
    """
    t = (text or "").strip()
    t = re.sub(r"(재직\s*증명서|재직증명서|증명서\s*출력|출력해|출력해줘|만들어줘|발급해|발급해줘)", "", t)
    t = t.strip()
    return t if t else None

def fetch_active_employees(name_hint: str | None = None, limit: int = 50) -> list[dict]:
    """
    현재 재직 중인 직원 리스트를 검색한다.
    name_hint(이름/사번 일부)에 따라 LIKE 검색도 가능하다.
    """
    where = """
    WHERE e.status = 'ACTIVE'
      AND (e.end_date IS NULL OR e.end_date > CURRENT_DATE)
    """
    params = {"limit": limit}

    if name_hint:
        where += " AND (e.emp_name ILIKE :q OR e.emp_id::text ILIKE :q)"
        params["q"] = f"%{name_hint}%"

    sql = f"""
    SELECT
      e.emp_id,
      e.emp_name,
      e.title,
      e.hire_date,
      e.email,
      d.dept_name
    FROM employees e
    LEFT JOIN departments d
      ON d.dept_id = e.dept_id
    {where}
    ORDER BY e.emp_name
    LIMIT :limit;
    """
    return fetch_all(sql, params)

# 한글 폰트(선택): 윈도우라면 보통 맑은 고딕 경로를 등록
def ensure_korean_font():
    """
    ReportLab에 한글 폰트(맑은 고딕)가 등록되지 않았으면 시스템 폰트 경로(윈도우 기준)에서 등록 시도.
    """
    try:
        pdfmetrics.getFont("MalgunGothic")
    except Exception:
        # 윈도우 기본 폰트 경로 (환경에 따라 다를 수 있음)
        font_path = r"C:\Windows\Fonts\malgun.ttf"
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("MalgunGothic", font_path))

# =====================================================
# 📄 재직증명서 PDF 생성
# =====================================================
def build_employment_certificate_pdf(emp: dict) -> bytes:
    """
    직원 dict 정보를 PDF 재직증명서로 생성해 bytes(다운로드/미리보기)로 반환하는 함수.
    """
    ensure_korean_font()  # 한글 폰트 등록 보장

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    FONT = "NotoSansKR"

    # 제목
    c.setFont(FONT, 22)
    c.drawCentredString(w / 2, h - 35 * mm, "재 직 증 명 서")
    c.line(20 * mm, h - 40 * mm, w - 20 * mm, h - 40 * mm)

    y = h - 60 * mm
    c.setFont(FONT, 12)

    def row(label, value):
        nonlocal y
        c.drawString(30 * mm, y, label)
        c.drawString(70 * mm, y, value)
        y -= 10 * mm

    hire = emp.get("hire_date")
    hire_str = hire.strftime("%Y-%m-%d") if isinstance(hire, (date, datetime)) else "-"

    row("성명", emp.get("emp_name", "-"))
    row("사번", str(emp.get("emp_id", "-")))
    row("부서", emp.get("dept_name", "-"))
    row("직위", emp.get("title", "-"))
    row("입사일", hire_str)
    row("재직상태", "재직 중")

    y -= 10 * mm
    c.drawString(30 * mm, y, "위 사람은 현재 당사에 재직 중임을 증명합니다.")

    y -= 25 * mm
    today = date.today().strftime("%Y년 %m월 %d일")
    c.drawRightString(w - 30 * mm, y, today)

    y -= 20 * mm
    c.drawRightString(w - 30 * mm, y, "주식회사 넝쿨HR")
    c.drawRightString(w - 30 * mm, y - 10, "대표이사 (인)")

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer.read()


FONT_PATH = "assets/fonts/NotoSansKR-Regular.ttf"
FONT_NAME = "NotoSansKR"

def ensure_korean_font():
    """
    ReportLab에서 사용할 한글 폰트가 등록되어 있지 않으면, 지정 경로의 폰트를 등록한다.
    """
    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"Font not found: {FONT_PATH}")
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))


@st.cache_data(show_spinner=False)
def _render_pdf_page_png(pdf_sha1: str, pdf_bytes: bytes, page_idx: int, zoom: float) -> bytes:
    """
    PDF 바이트와 페이지 인덱스를 받아 PNG 바이트로 렌더링
    - 동일 pdf/페이지/확대비율이면 바로 캐시 사용
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc.load_page(int(page_idx))
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()

def pdf_preview(pdf_bytes: bytes, default_zoom: float = 1.4):
    """
    Streamlit에서 PDF를 이미지로 미리보기 렌더링하는 함수.
    페이지 전환, 확대, 폭맞춤 토글 등 ui 컨트롤 포함
    """
    if not pdf_bytes:
        return

    # 캐시 키로 쓸 sha1 해시값 계산
    pdf_sha1 = hashlib.sha1(pdf_bytes).hexdigest()

    # 페이지 수는 한번만 체크(캐시 이외)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = doc.page_count
    finally:
        doc.close()

    ctrl_col, view_col = st.columns([1, 5], vertical_alignment="top")

    with ctrl_col:
        st.markdown("#### 🔍 보기 설정")

        if page_count > 1:
            page_idx = st.number_input("페이지", 1, page_count, 1) - 1
        else:
            page_idx = 0

        zoom = st.slider("확대", 0.8, 3.0, float(default_zoom), 0.05)
        fit_to_width = st.toggle("화면에 맞춤", value=True)

    # PDF -> 이미지 렌더
    png_bytes = _render_pdf_page_png(pdf_sha1, pdf_bytes, int(page_idx), float(zoom))
    img = Image.open(BytesIO(png_bytes))

    with view_col:
        if fit_to_width:
            st.image(img, use_container_width=True)
        else:
            st.image(img, use_container_width=False)

# =====================================================
# 1) 페이지 설정 / 세션
# =====================================================
st.set_page_config(page_title="Agentic AI for 넝쿨HR", layout="wide")

# 다양한 세션 변수(메시지, 질문, 시나리오 등) 초깃값 세팅
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



# =====================================================
# CSS (상단 공백 제거)
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
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# 2) 환경변수 로드
# ===============================
def get_google_api_key() -> str | None:
    """
    환경변수에서 GOOGLE_API_KEY 값을 읽어온다.
    """
    return os.getenv("GOOGLE_API_KEY")


def get_db_uri() -> str | None:
    """
    환경변수에서 SUPABASE_DB_URI 값을 읽어온다.
    """
    return os.getenv("SUPABASE_DB_URI")

api_key = get_google_api_key()
db_uri = get_db_uri()

# ===============================
# 3) 환경변수 검증
# ===============================
# LLM API KEY, DB URI 미설정 시 안내 후 앱 중단
if not api_key:
    st.error("❌ GOOGLE_API_KEY가 설정되어 있지 않습니다. (Render: Environment Variables 확인)")
    st.stop()

if not db_uri:
    st.error("❌ SUPABASE_DB_URI이 설정되어 있지 않습니다. (Render: Environment Variables 확인)")
    st.stop()

if "YOUR-PASSWORD" in db_uri:
    st.error("❌ SUPABASE_DB_URI에 [YOUR-PASSWORD]가 그대로 있습니다.")
    st.stop()


# =====================================================
# 3) HR/LLM 엔진 + Explainer
# =====================================================
@st.cache_resource(show_spinner=False)
def get_hr_engine(_db_uri: str, _api_key: str, _version: str) -> HRTextToSQLEngine:
    """
    HRTextToSQLEngine (LLM SQL 생성+실행 엔진)를 환경값에 맞춰 한 번만 생성 (캐시).
    """
    return HRTextToSQLEngine(db_uri=_db_uri, api_key=_api_key)


def ensure_hr_engine() -> HRTextToSQLEngine:
    """
    HRTextToSQLEngine 인스턴스를 캐시에서 불러오기. 필요시만 호출.
    """
    return get_hr_engine(db_uri, api_key, ENGINE_VERSION)

@st.cache_resource(show_spinner=False)
def get_explainer(_api_key: str):
    """
    SQL 실행결과를 한글로 명확히 해설/요약해주는 Gemini 기반 체인 반환.
    """
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
    사용자의 불완전한 질문을 대화 히스토리를 참고해 '완전한 독립문장'으로 재작성(프롬프트)해주는 체인 반환.
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

@st.cache_resource(show_spinner=False)
def get_decision_classifier(_api_key: str):
    prompt = ChatPromptTemplate.from_template(
        """
You are an HR Decision Type Classifier.

Your task is to classify the user's question into one of:
- DATA_QUERY: asking only for information or facts
- DECISION: asking whether an action should be taken
- MIXED: asking for data AND what decision/action to take

IMPORTANT RULES:
- Questions that ask whether something should be done
  (e.g. "해야 할까", "뽑아야 할까", "늘려야 할까", "줄여야 할까", "필요할까")
  MUST be classified as DECISION.
- Requests for advice, judgment, recommendation, or evaluation are DECISION.
- Only pure requests for data, lists, or numbers are DATA_QUERY.

HR Decision Types (use only when intent is DECISION or MIXED):
- STAFFING
- WORKLOAD
- COMPENSATION
- PERFORMANCE
- LEAVE
- ORG_STRUCTURE
- POLICY

Output MUST be a JSON object with exactly these keys:
{{
  "intent": "DATA_QUERY | DECISION | MIXED",
  "decision_type": "STAFFING | WORKLOAD | COMPENSATION | PERFORMANCE | LEAVE | ORG_STRUCTURE | POLICY | null"
}}

Examples:

Input: 마케팅팀 인원 더 뽑아야 할까?
Output:
{{
  "intent": "DECISION",
  "decision_type": "STAFFING"
}}

Input: 요즘 야근이 너무 많은 것 같아
Output:
{{
  "intent": "DECISION",
  "decision_type": "WORKLOAD"
}}

Input: 이번 달 부서별 평균 근무시간은?
Output:
{{
  "intent": "DATA_QUERY",
  "decision_type": null
}}

Now classify the following input.

Input: {question}
"""
    )

    return (
        prompt
        | ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=_api_key,
            temperature=0.0
        )
        | StrOutputParser()
    )


DECISION_ACTION_TEMPLATES = {
    "STAFFING": [
        "현재 인원 현황 보여줘",
        "최근 3개월 업무량 추이 보여줘",
        "최근 이직률 높은 팀은?"
    ],
    "WORKLOAD": [
        "부서별 평균 근무시간 보여줘",
        "야근 많은 팀 TOP 5",
        "최근 한달 업무량 변화는?"
    ],
    "COMPENSATION": [
        "직급별 평균 연봉 보여줘",
        "최근 이직자 보상 수준은?",
        "팀별 연봉 편차 보여줘"
    ],
    "PERFORMANCE": [
        "팀별 성과 지표 요약해줘",
        "성과 낮은 팀은 어디야?",
        "최근 평가 결과 분포는?"
    ],
    "LEAVE": [
        "부서별 휴가 사용률 보여줘",
        "휴가 사용 적은 팀은?",
        "승인 대기 중인 휴가 요청은?"
    ],
    "ORG_STRUCTURE": [
        "팀별 인원 구성 보여줘",
        "관리자 1인당 인원수는?",
        "조직 구조 요약해줘"
    ],
    "POLICY": [
        "현재 HR 정책 목록 보여줘",
        "최근 정책 변경 이력은?",
        "정책별 적용 대상은?"
    ]
}

def format_history(messages, limit=6):
    """
    세션에 저장된 메시지 중 최근 N개를 user/assistant 구분과 함께 텍스트로 변환(이상형 대화 이력 string).
    너무 오래된 것은 잘라내고 최근 limit개 정도만 반환한다.
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
    Gemini/QuerySQLDatabaseTool 등에서 SQL 결과가 list/tuple, 문자열등 여러 형태로 들어오므로 
    일관적으로 list 결과(딕셔너리 or 튜플)로 변환해서 반환.
    """
    if result is None:
        return []

    if isinstance(result, (list, tuple)):
        return list(result)

    if isinstance(result, str):
        s = result.strip()
        # Decimal, UUID 등 문자열을 파이썬 기본타입으로 치환하여 파싱
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
    HRTextToSQLEngine의 executor를 활용해(캐시엔진 활용) SQL을 실행하고 결과 반환.
    """
    hr = ensure_hr_engine()
    return hr.executor.invoke({"query": sql})


def classify_decision(question: str) -> dict:
    classifier = get_decision_classifier(api_key)

    raw = classifier.invoke({"question": question})

    # markdown fence 제거
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except Exception as e:
        return {
            "intent": "DATA_QUERY",
            "decision_type": None,
            "error": str(e),
            "raw": raw
        }


def fmt_won(n):
    """
    숫자를 세 자리 콤마와 '원' 단위로 출력 (에러시 그대로 리턴)
    예: 1000000 -> 1,000,000원
    """
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
    """
    질문에서 2026-01, 2026년 1월 등 '년-월' 기간을 추출
    """
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
    """
    질문 텍스트에서 '전체/전직원/부서 등' 범위(scope) 지정 키워드 추출
    """
    t = text.strip()

    if re.search(r"(전\s*직원|전체\s*직원|전체|전사|모두|전부서|전\s*부서|전부\s*서)", t):
        return "ALL"

    m = re.search(r"\b([가-힣A-Za-z0-9_]+)\s*(부|팀)\b", t)
    if m:
        return f"dept:{m.group(1)}{m.group(2)}"

    return None


def extract_date_any(text: str):
    """
    yyyy-mm-dd, m/d, 일 등 날짜 관련 정보 패턴을 찾아 date string으로 반환(년은 period로 유추)
    """
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
    """
    예/아니오/확정/취소 등 사용자의 확인(확정의도) 값을 True/False/None으로 해석
    """
    t = text.strip()
    if re.fullmatch(r"(예|네|응|진행|실행|확정|ok|ㅇㅋ)", t, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"(아니오|아니|취소|중단|no|ㄴㄴ)", t, flags=re.IGNORECASE):
        return False
    return None


def is_rpc_trigger(text: str):
    """
    급여/공제/전표 등 RPC 실행 모드용 키워드가 들어있으면 True
    """
    return bool(re.search(r"(급여|세금|공제|지급|이체|송금|전표|분개)", text)) and (
        is_execute_intent(text) or not is_query_intent(text)
    )


def is_execute_intent(text: str) -> bool:
    """
    실질적인 실행 의도(계산, 처리, 전표 생성 등)가 있는 질문이면 True
    """
    t = text.strip()
    return bool(re.search(r"(처리|실행|진행|계산|산정해|돌려|생성해|등록|전표생성|지급해)", t))


def is_query_intent(text: str) -> bool:
    """
    조회 의도(총액, 대장, 내역 등)가 포함된 질문인지 판별
    """
    t = text.strip()
    return bool(re.search(r"(몇\s*명|인원|대상|총액|합계|금액|건수|결과|내역|리스트|상세|조회|보여줘)", t))


def month_to_period_date(period_yyyy_mm: str):
    """
    '2026-01' 등 year-month를 '2026-01-01' 등 y-m-1 포맷으로 변환.
    """
    y, m = period_yyyy_mm.split("-")
    return f"{int(y):04d}-{int(m):02d}-01"


# =====================================================
# 6) RPC 시나리오 오케스트레이터(최소)
# =====================================================
# 주요 status/state 값과 화면 표시용 LABEL 매핑
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

# 시나리오 상태(메모리) 관리를 위한 래퍼
memory = ScenarioMemoryManager(store=st.session_state, namespace="scenario_memory")


def rpc_get_ctx(session_id: str) -> dict:
    """
    세션별 RPC 시나리오 컨텍스트 상태(dict) 읽기 (없으면 빈 dict)
    """
    return memory.get(session_id) or {}


def rpc_set_ctx(session_id: str, ctx: dict):
    """
    세션별 RPC 시나리오 컨텍스트 저장(갱신)
    """
    memory.set(session_id, ctx)


def rpc_clear_ctx(session_id: str):
    """
    세션별 RPC 시나리오 상태/메모리 초기화(삭제)
    """
    memory.clear(session_id)


def rpc_fetch_run(run_id: str):
    """
    process_runs 테이블에서 단일 run_id의 기록(상태, 요약 등)을 조회하고 sql 문자열도 같이 반환.
    """
    sql = f"""
    select run_id, process_type, period, scope, status, params, summary, error_msg, started_at, finished_at
    from public.process_runs
    where run_id = '{run_id}';
    """
    return exec_sql(sql), sql.strip()


def rpc_fetch_lines(run_id: str):
    """
    process_run_lines 테이블에서 특정 배치(run)의 라인(세부 지급/전표 행)들을 조회.
    """
    sql = f"""
    select line_id, line_type, data, created_at
    from public.process_run_lines
    where run_id = '{run_id}'
    order by line_id;
    """
    return exec_sql(sql), sql.strip()


def rpc_answer_query_from_refs(ctx: dict, user_text: str):
    """
    시나리오 context(refs)에 직전 run_id들이 남아 있다면,
    사용자의 조회형 질문(user_text)에 맞는 정보를 즉답해주는 함수 (예: '전표 라인 몇 건?')
    """
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

    # 인원수 조회
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

    # 그 외에는 요약 내용 전체 전달
    return {"reply": f"📌 요약: {summary}", "sqls": [sql_fetch]}


def rpc_run(session_id: str, user_text: str) -> dict:
    """
    급여~전표 각 단계별로 조건, 확인 등을 체크하며
    각 시나리오 진행을 담당하는 오케스트레이터 함수. 상태기반 분기/실행
    """
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
        """
        __MD__ 형식 등 약식 날짜를 yyyy-mm-dd로 변환
        """
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

    # 이하 단계별 긴 분기(급여 산정, 공제, 지급, 전표, 완료)는 기존처럼 주석 생략 (상세 설명은 위 안내 참고)
    # state별 블록 내부 로직에는 주석이 있으니 생략 (중복될 우려 있음!)

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
# 서비스를 소개하는 상단 헤더/설명 표시 마크다운 렌더
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
# 6.5) 왼쪽 사이드바 - Agentic HR 설정
# =====================================================
with st.sidebar:
    st.markdown("### ⚙️ Agentic HR 설정")

    if st.button("🗑️ 대화 기록 지우기", key="sidebar_clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.action_suggestions = []
        st.session_state.pending_question = None
        st.rerun()

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
# 현재 RPC 모드 활성시 상태바(단계, 종료버튼) 표시
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
# 추천 질문(칩) UI 표시 & 클릭 시 질문 입력란에 자동 반영
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
# 전체 메시지를 user/assistant 기준 turn별로 화면에 채팅 UI로 표시
turns = build_turns(st.session_state.messages)

# 마지막으로 SQL 쿼리가 실행된 턴 찾기(설명 열림 표시용)
last_sql_turn_idx = -1
for i, t in enumerate(turns):
    a = t.get("assistant") or {}
    if a.get("sql") or a.get("raw_sql"):
        last_sql_turn_idx = i

def _render_pdf_file_preview(file_path):
    """
    파일 경로에서 PDF를 읽어 미리보기와 다운로드 버튼을 렌더링하는 함수
    """
    if not file_path or not os.path.exists(file_path):
        st.warning("⚠️ PDF 파일을 찾을 수 없습니다.")
        return
    
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_preview(pdf_bytes)
        st.download_button(
            "⬇️ PDF 다운로드",
            data=pdf_bytes,
            file_name=os.path.basename(file_path),
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"❌ PDF 파일을 읽는 중 오류가 발생했습니다: {e}")

for i, t in enumerate(turns):
    if t["user"]:
        with st.chat_message("user"):
            agent_progress = t["user"].get("agent_progress", None)
            if t["user"].get("show_agent_progress") and agent_progress:
                with st.expander("🤖 에이전트 처리 단계", expanded=True):
                    for step in agent_progress:
                        label = step.get("label", "")
                        status = step.get("status", "")
                        if status == "doing":
                            with st.status(f"{label} 처리중...", expanded=True):
                                pass
                        elif status == "done":
                            st.success(f"{label} 완료")
                        elif status == "error":
                            st.error(f"{label} 실패")
                        else:
                            st.info(f"{label}")
            st.markdown(t["user"]["content"])

    if t["assistant"]:
        with st.chat_message("assistant"):
            agent_progress = t["assistant"].get("agent_progress", None)
            if agent_progress:
                with st.expander("🤖 에이전트 처리 단계", expanded=True):
                    for step in agent_progress:
                        label = step.get("label", "")
                        status = step.get("status", "")
                        if status == "doing":
                            with st.status(f"{label} 처리중...", expanded=True):
                                pass
                        elif status == "done":
                            st.success(f"{label} 완료")
                        elif status == "error":
                            st.error(f"{label} 실패")
                        else:
                            st.info(f"{label}")
            st.markdown(t["assistant"]["content"])
            expand_this = (i == last_sql_turn_idx)

            # assistant 메시지에 file_path가 있으면 해당 말풍선 밑에 미리보기(expander) 렌더링
            file_path = t["assistant"].get("file_path")
            if file_path:
                with st.expander("📄 첨부: 재직증명서", expanded=True):
                    _render_pdf_file_preview(file_path)

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
# 12) 실행: (재직증명서 트리거 우선) → (RPC 실행 모드) → fallback LLM 조회
# =====================================================

if question:
    # user 메시지 기록
    st.session_state.messages.append({"role": "user", "content": question})

    # =====================================================
    # (0.5) 🧠 Decision Type Classifier
    # =====================================================
    decision_notice = None
    decision_actions = []

    decision = classify_decision(question)
    intent = decision.get("intent")
    decision_type = decision.get("decision_type")

    execute_mode = st.session_state.get("rpc_execute_mode", False)

    if intent == "DECISION" and not execute_mode:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"🧠 이 질문은 **{decision_type} 관련 의사결정**으로 인식했어요.\n\n"
                "바로 결론을 내리기보다는, 판단에 필요한 근거부터 확인해볼게요."
            )
        })

        st.session_state.action_suggestions = (
            DECISION_ACTION_TEMPLATES.get(decision_type, [])
        )

        request_scroll("result-anchor")
        st.rerun()   # 🔥 핵심: rerun으로 렌더링 트리거

    # =====================================================
    # 결과 변수 초기화
    # =====================================================
    answer = ""
    sql_to_show = None
    raw_sql_to_show = None
    file_path_to_save = None

    try:
        # =====================================================
        # (0) 📄 재직증명서 트리거 우선 처리
        # =====================================================
        if is_employment_cert_trigger(question):
            with st.spinner("재직증명서 조회 중..."):
                name_hint = extract_employee_hint(question)
                employees = fetch_active_employees(name_hint=name_hint, limit=50)

            if not employees:
                answer = "❌ 재직 중인 직원을 찾지 못했습니다. 이름/사번을 포함해서 다시 입력해 주세요."
            else:
                options = {
                    f"{(e.get('emp_name') or e.get('name'))} ({e.get('dept_name','-')}, {e.get('emp_id')})": e
                    for e in employees
                }

                if len(options) == 1:
                    selected = list(options.values())[0]
                else:
                    st.info("재직증명서를 발급할 직원을 선택해 주세요.")
                    label = st.selectbox("직원 선택", list(options.keys()), key="employment_select")
                    selected = options[label]

                with st.spinner("재직증명서 PDF 생성 중..."):
                    pdf_bytes = build_employment_certificate_pdf(selected)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
                    tmpf.write(pdf_bytes)
                    file_path_to_save = tmpf.name

                emp_display = selected.get("emp_name") or "직원"
                answer = f"📄 **{emp_display}** 님 재직증명서를 생성했습니다. 아래에서 확인하세요."

            msg = {"role": "assistant", "content": answer}
            if file_path_to_save:
                msg["file_path"] = file_path_to_save

            st.session_state.messages.append(msg)
            request_scroll("result-anchor")
            st.rerun()

        # =====================================================
        # (1) 실행 모드: RPC
        # =====================================================
        if execute_mode:
            with st.spinner("처리 중..."):
                out_rpc = rpc_run(st.session_state.session_id, question)

            answer = out_rpc.get("reply", "")
            st.session_state.action_suggestions = out_rpc.get("suggestions", []) or []

            rpc_sqls = (out_rpc.get("artifacts", {}) or {}).get("rpc_sqls", [])
            if rpc_sqls:
                sql_to_show = "\n\n".join(s.strip() for s in rpc_sqls)

        # =====================================================
        # (2) 조회 모드: LLM SQL 조회
        # =====================================================
        else:
            with st.spinner("처리 중... (질문 해석 → SQL 생성 → 실행 → 요약)"):
                hr = ensure_hr_engine()

                real_question = question
                if len(st.session_state.messages) > 1:
                    rewriter = get_rewriter(api_key)
                    history_str = format_history(st.session_state.messages[:-1])
                    real_question = rewriter.invoke({
                        "history": history_str,
                        "question": question
                    })

                out = hr.run(real_question)
                fixed_sql = out.get("fixed_sql") or ""
                raw_sql = out.get("raw_sql")

                patched_sql = enforce_month_range_sql(fixed_sql)
                patched_result = exec_sql(patched_sql)

                answer_body = explainer.invoke({
                    "question": real_question,
                    "result": patched_result
                })

                if decision_notice:
                    answer = decision_notice + answer_body
                    st.session_state.action_suggestions = decision_actions
                else:
                    answer = answer_body
                    st.session_state.action_suggestions = []

                sql_to_show = patched_sql
                raw_sql_to_show = fixed_sql if raw_sql is None else raw_sql

    except Exception as e:
        answer = f"❌ 오류: {e}"
        st.session_state.action_suggestions = []

    # =====================================================
    # assistant 메시지 최종 1회 append
    # =====================================================
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sql": sql_to_show,
        "raw_sql": raw_sql_to_show,
    })

    request_scroll("result-anchor")
    st.rerun()

