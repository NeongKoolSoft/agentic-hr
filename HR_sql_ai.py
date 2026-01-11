import re
import uuid
from typing import Optional

from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from scenario_payroll import (
    ScenarioMemoryManager,
    PayrollScenario,
    ScenarioOrchestrator,
)

ENGINE_VERSION = "v2026-01-08-05"  # ✅ 시나리오 통합 버전

# =====================================================
# 0) SQL 정제 및 보정
# =====================================================
def strip_code_fence(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def normalize_sql(s: str) -> str:
    s = strip_code_fence(s)
    s = re.sub(
        r"^\s*(sql|postgres|postgresql)\s*:?\s*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    m = re.search(r"\b(select|with)\b", s, flags=re.IGNORECASE)
    if m:
        s = s[m.start():].strip()

    return s.rstrip(";").strip() + ";"


def fix_postgres_date_sql(sql: str) -> str:
    sql = strip_code_fence(sql).strip()

    sql = re.sub(r"'(\d{4}-\d{2}-\d{2})'+", r"'\1'", sql)
    sql = re.sub(r"\bDATE\s+DATE\b", "DATE", sql, flags=re.IGNORECASE)
    sql = re.sub(
        r"\bDATE\s+(\d{4}-\d{2}-\d{2})",
        r"DATE '\1'",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"\bINTERVAL\s+(\d+)\s+(day|days)",
        r"INTERVAL '\1 \2'",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"INTERVAL\s+'([^']*)'+",
        r"INTERVAL '\1'",
        sql,
        flags=re.IGNORECASE,
    )

    return sql


def is_safe_readonly_sql(sql: str) -> bool:
    if not sql:
        return False

    lowered = sql.lower().strip()

    if len([p for p in lowered.split(";") if p.strip()]) != 1:
        return False

    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False

    forbidden = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "pg_",
    ]
    return not any(tok in lowered for tok in forbidden)


# =====================================================
# 1) SQL 프롬프트
# =====================================================
SQL_PROMPT = PromptTemplate.from_template(
    """당신은 HR 데이터베이스를 다루는 PostgreSQL 전문가입니다.

규칙:
- 반드시 단 하나의 SELECT 문만 생성하십시오.
- 오늘 날짜는 DATE '2026-01-08'로 고정해서 사용하십시오.
- CURRENT_DATE 대신 반드시 DATE '2026-01-08'을 사용하십시오.
- 날짜 연산 예시: DATE '2026-01-08' - INTERVAL '30 days'
- SQL 외의 설명, 주석, 마크다운은 절대 포함하지 마십시오.

스키마:
{schema}

질문:
{question}

SQL:"""
)

# =====================================================
# 2) Text → SQL 엔진
# =====================================================
class HRTextToSQLEngine:
    def __init__(self, db_uri: str, api_key: str):
        self.db = SQLDatabase.from_uri(db_uri)
        self.executor = QuerySQLDatabaseTool(db=self.db)

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0,
        )

        self._schema: Optional[str] = None

        self.chain = (
            {"schema": lambda _: self.schema, "question": RunnablePassthrough()}
            | SQL_PROMPT
            | self.llm
            | StrOutputParser()
        )

    @property
    def schema(self) -> str:
        if self._schema is None:
            self._schema = self.db.get_table_info()
        return self._schema

    def run(self, question: str) -> dict:
        raw_sql = self.chain.invoke(question)

        sql = normalize_sql(raw_sql)
        sql = fix_postgres_date_sql(sql)

        if not is_safe_readonly_sql(sql):
            raise ValueError(f"위험한 쿼리 차단됨: {sql}")

        try:
            result = self.executor.invoke({"query": sql})
            return {
                #"raw_sql": raw_sql,
                "fixed_sql": sql,
                "result": result,
            }
        except Exception as e:
            return {
                #"raw_sql": raw_sql,
                "fixed_sql": sql,
                "error": str(e),
            }


# =====================================================
# 3) Scenario + Orchestrator 생성 헬퍼
# =====================================================
def build_orchestrator(
    db_uri: str,
    api_key: str,
    session_store: dict,
):
    """
    UI 레이어에서 한 번만 호출해서 orchestrator를 만들어두고
    orchestrator.run(session_id, user_text) 형태로 사용
    """
    engine = HRTextToSQLEngine(db_uri, api_key)

    memory = ScenarioMemoryManager(
        store=session_store,
        namespace="scenario_memory",
    )


    payroll_scenario = PayrollScenario(memory)

    orchestrator = ScenarioOrchestrator(
        sql_engine=engine,
        scenarios=[payroll_scenario],
    )

    return orchestrator
