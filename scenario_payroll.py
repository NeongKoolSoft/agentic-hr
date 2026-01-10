# scenario_payroll.py
# Minimal scaffolding to plug next to HRTextToSQLEngine
# - ScenarioMemoryManager: session-based memory (in-memory; can be backed by st.session_state)
# - PayrollScenario: "급여 → 공제검증 → 지급 → 전표" state machine with slot filling + confirm gates

from __future__ import annotations

import re
import uuid
import ast
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional, List, Tuple


# -----------------------------
# Result parsing helpers
# -----------------------------
def _to_rows(result):
    """
    QuerySQLDatabaseTool 결과는 보통 문자열로 옵니다.
    예) "[(26, Decimal('92082741'), Decimal('7611337'), 0)]"
    Decimal(...) 같은 표현 때문에 literal_eval이 실패할 수 있어서 정규화 후 파싱합니다.
    """
    if result is None:
        return []

    if isinstance(result, (list, tuple)):
        return list(result)

    if isinstance(result, str):
        s = result.strip()

        # Decimal('123') -> 123
        s = re.sub(r"Decimal\('(-?\d+(?:\.\d+)?)'\)", r"\1", s)
        # Decimal("123") -> 123
        s = re.sub(r'Decimal\("(-?\d+(?:\.\d+)?)"\)', r"\1", s)
        # NULL -> None
        s = s.replace("NULL", "None")

        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]
        except Exception:
            # fallback: numeric tokens
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if nums:
                row = []
                for x in nums:
                    row.append(int(x) if re.fullmatch(r"-?\d+", x) else float(x))
                return [tuple(row)]
            return []

    return []


def _fmt_won(n):
    try:
        return f"{int(float(n)):,}원"
    except Exception:
        return str(n)


# -----------------------------
# 0) Memory Manager
# -----------------------------
class ScenarioMemoryManager:
    """
    Very small memory store keyed by session_id.
    - default store: in-memory dict
    - you can pass streamlit.session_state (dict-like) as store
      e.g. ScenarioMemoryManager(store=st.session_state)
    """
    def __init__(self, store: Optional[dict] = None, namespace: str = "scenario_mem"):
        self._external_store = store
        self._namespace = namespace
        if self._external_store is None:
            self._mem: Dict[str, dict] = {}
        else:
            if namespace not in self._external_store:
                self._external_store[namespace] = {}
            self._mem = self._external_store[namespace]

    def get(self, session_id: str) -> dict:
        return self._mem.get(session_id, {})

    def set(self, session_id: str, data: dict) -> None:
        self._mem[session_id] = data

    def clear(self, session_id: str) -> None:
        if session_id in self._mem:
            del self._mem[session_id]


# -----------------------------
# 1) Scenario Model
# -----------------------------
TODAY = date(2026, 1, 8)  # align with your prompt rule

STATE_PAYROLL_CALC = "PAYROLL_CALC"
STATE_TAX_CALC = "TAX_CALC"         # 실제 의미: 공제 검증/요약
STATE_PAYMENT_RUN = "PAYMENT_RUN"
STATE_JOURNAL_POST = "JOURNAL_POST"
STATE_DONE = "DONE"

ACTIVE_SCENARIO = "PAYROLL_E2E"


@dataclass
class ScenarioContext:
    active_scenario: str = ""
    state: str = ""
    slots: Dict[str, Any] = field(default_factory=dict)
    refs: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "active_scenario": self.active_scenario,
            "state": self.state,
            "slots": self.slots,
            "refs": self.refs,
            "history": self.history,
        }

    @staticmethod
    def from_dict(d: dict) -> "ScenarioContext":
        return ScenarioContext(
            active_scenario=d.get("active_scenario", ""),
            state=d.get("state", ""),
            slots=dict(d.get("slots", {}) or {}),
            refs=dict(d.get("refs", {}) or {}),
            history=list(d.get("history", []) or []),
        )


# -----------------------------
# 2) Slot Extractor (minimal rule-based)
# -----------------------------
def _extract_period(text: str) -> Optional[str]:
    t = text.strip()

    # yyyy-mm
    m = re.search(r"\b(20\d{2})[-./](0?[1-9]|1[0-2])\b", t)
    if m:
        y = m.group(1)
        mm = int(m.group(2))
        return f"{y}-{mm:02d}"

    # yyyy년 n월
    m = re.search(r"\b(20\d{2})\s*년\s*(0?[1-9]|1[0-2])\s*월\b", t)
    if m:
        y = m.group(1)
        mm = int(m.group(2))
        return f"{y}-{mm:02d}"

    # n월 (assume current year in TODAY)
    m = re.search(r"\b(0?[1-9]|1[0-2])\s*월\b", t)
    if m:
        mm = int(m.group(1))
        return f"{TODAY.year}-{mm:02d}"

    if re.search(r"(이번\s*달|당월|이번달)", t):
        return f"{TODAY.year}-{TODAY.month:02d}"

    if re.search(r"(지난\s*달|전월|지난달)", t):
        y, mth = TODAY.year, TODAY.month - 1
        if mth == 0:
            y -= 1
            mth = 12
        return f"{y}-{mth:02d}"

    return None


def _extract_employee_scope(text: str) -> Optional[str]:
    t = text.strip()

    if re.search(r"(전\s*직원|전체\s*직원|전체|전사|모두)", t):
        return "ALL"

    m = re.search(r"\b([가-힣A-Za-z0-9_]+)\s*(부|팀)\b", t)
    if m:
        return f"dept:{m.group(1)}{m.group(2)}"

    m = re.search(r"\b([가-힣]{2,4})\b", t)
    if m and not re.search(r"(급여|세금|지급|전표|이번|지난|월|년|공제)", m.group(1)):
        return f"emp:{m.group(1)}"

    return None


def _extract_pay_date(text: str) -> Optional[str]:
    t = text.strip()

    m = re.search(r"\b(20\d{2})[-./](0?[1-9]|1[0-2])[-./](0?[1-9]|[12]\d|3[01])\b", t)
    if m:
        y = int(m.group(1))
        mm = int(m.group(2))
        dd = int(m.group(3))
        return f"{y}-{mm:02d}-{dd:02d}"

    m = re.search(r"\b(0?[1-9]|[12]\d|3[01])\s*일\b", t)
    if m and re.search(r"(지급|이체|송금)", t):
        return f"__DAY__:{int(m.group(1))}"

    return None


def _extract_confirm(text: str) -> Optional[bool]:
    t = text.strip()
    if re.fullmatch(r"(예|네|응|진행|실행|확정|ok|ㅇㅋ)", t, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"(아니오|아니|취소|중단|no|ㄴㄴ)", t, flags=re.IGNORECASE):
        return False
    return None


def extract_slots(text: str) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}
    p = _extract_period(text)
    if p:
        slots["period"] = p

    scope = _extract_employee_scope(text)
    if scope:
        slots["employee_scope"] = scope

    pay_date = _extract_pay_date(text)
    if pay_date:
        slots["pay_date"] = pay_date

    conf = _extract_confirm(text)
    if conf is not None:
        slots["confirm"] = conf

    # intent flags
    if re.search(r"(급여|급여처리|급여\s*계산|급여\s*산정)", text):
        slots["intent"] = "PAYROLL"
    if re.search(r"(세금|원천세|4대보험|보험료|공제|공제금액)", text):
        slots["intent"] = "TAX"
    if re.search(r"(지급|이체|송금)", text):
        slots["intent"] = "PAYMENT"
    if re.search(r"(전표|분개|전기|회계)", text):
        slots["intent"] = "JOURNAL"
    if re.search(r"(취소|종료|그만|중단|처음부터|리셋|초기화)", text):
        slots["intent"] = "EXIT"

    return slots


# -----------------------------
# 3) Payroll Scenario
# -----------------------------
class PayrollScenario:
    def __init__(self, memory: ScenarioMemoryManager):
        self.memory = memory

    def route_and_handle(self, session_id: str, user_text: str, sql_engine) -> dict:
        slots = extract_slots(user_text)
        ctx = ScenarioContext.from_dict(self.memory.get(session_id))

        is_trigger = self._is_payroll_trigger(user_text, slots, ctx)
        if not is_trigger:
            return {"handled": False}

        # activate scenario if not active
        if ctx.active_scenario != ACTIVE_SCENARIO:
            ctx = ScenarioContext(
                active_scenario=ACTIVE_SCENARIO,
                state=STATE_PAYROLL_CALC,
                slots={},
                refs={},
                history=[],
            )

        # keep last text (journal date parsing etc)
        ctx.slots["_last_user_text"] = user_text

        # EXIT
        if slots.get("intent") == "EXIT":
            self.memory.clear(session_id)
            return {
                "handled": True,
                "reply": "급여 시나리오를 종료하고 초기 상태로 돌아왔습니다. 다른 질문을 해주세요.",
                "state": None,
                "suggestions": [],
            }

        # merge new slots (except intent/confirm)
        ctx.slots.update({k: v for k, v in slots.items() if k not in ("intent", "confirm")})
        confirm = slots.get("confirm", None)

        self._normalize_slots(ctx)

        # allow direct jump (but prerequisites guard는 각 state에서 처리)
        intent = slots.get("intent")
        if intent:
            self._maybe_jump_state(ctx, intent)

        out = self._handle_state(ctx, sql_engine, confirm)

        self.memory.set(session_id, ctx.to_dict())
        return out

    def _is_payroll_trigger(self, text: str, slots: Dict[str, Any], ctx: ScenarioContext) -> bool:
        if ctx.active_scenario == ACTIVE_SCENARIO:
            return True
        return bool(re.search(r"(급여|원천세|4대보험|지급|전표|공제)", text))

    def _maybe_jump_state(self, ctx: ScenarioContext, intent: str) -> None:
        if intent == "PAYROLL":
            ctx.state = STATE_PAYROLL_CALC
        elif intent == "TAX":
            ctx.state = STATE_TAX_CALC
        elif intent == "PAYMENT":
            ctx.state = STATE_PAYMENT_RUN
        elif intent == "JOURNAL":
            ctx.state = STATE_JOURNAL_POST

    def _normalize_slots(self, ctx: ScenarioContext) -> None:
        pay_date = ctx.slots.get("pay_date")
        if isinstance(pay_date, str) and pay_date.startswith("__DAY__:"):
            if "period" in ctx.slots:
                day = int(pay_date.split(":")[1])
                y, m = ctx.slots["period"].split("-")
                ctx.slots["pay_date"] = f"{int(y):04d}-{int(m):02d}-{day:02d}"

    # ---- state handling ----
    def _handle_state(self, ctx: ScenarioContext, sql_engine, confirm: Optional[bool]) -> dict:
        if ctx.state == STATE_PAYROLL_CALC:
            return self._state_payroll_calc(ctx, sql_engine)

        if ctx.state == STATE_TAX_CALC:
            return self._state_tax_calc(ctx, sql_engine)

        if ctx.state == STATE_PAYMENT_RUN:
            return self._state_payment_run(ctx, sql_engine, confirm)

        if ctx.state == STATE_JOURNAL_POST:
            return self._state_journal_post(ctx, sql_engine, confirm)

        if ctx.state == STATE_DONE:
            return self._state_done(ctx)

        ctx.state = STATE_PAYROLL_CALC
        return self._state_payroll_calc(ctx, sql_engine)

    def _require(self, ctx: ScenarioContext, keys: List[str]) -> Tuple[bool, List[str]]:
        missing = [k for k in keys if not ctx.slots.get(k)]
        return (len(missing) == 0, missing)

    def _make_run_id(self, prefix: str, period: str) -> str:
        return f"{prefix}_{period.replace('-', '')}_{uuid.uuid4().hex[:6].upper()}"

    # -----------------------------
    # STATE 1) PAYROLL_CALC
    # -----------------------------
    def _state_payroll_calc(self, ctx: ScenarioContext, sql_engine) -> dict:
        ok, missing = self._require(ctx, ["period", "employee_scope"])
        if not ok:
            reply = (
                f"급여 산정을 위해 정보가 필요합니다: {', '.join(missing)}\n"
                f"- 예: '2026년 1월 전 직원 급여 산정(미리보기)'\n"
                f"- 예: '1월 영업부 급여 계산'"
            )
            return {
                "handled": True,
                "reply": reply,
                "state": ctx.state,
                "suggestions": ["2026-01 전 직원 급여 산정", "이번달 전 직원 급여 산정"],
            }

        period = ctx.slots["period"]
        scope = ctx.slots["employee_scope"]
        calc_mode = ctx.slots.get("calc_mode") or "preview"

        question = (
            f"{period} 급여 산정(집계) {calc_mode}를 해줘. "
            f"대상은 {scope}야. "
            f"결과는 (대상 인원수, 총급여(기본+수당), 총공제액(deductions), 누락/오류 건수) 1행으로 요약해줘."
        )

        # ✅ 실행 결과를 res로 받고, 파싱도 res로 한다
        res = sql_engine.run(question)

        # ✅ 최신 artifacts를 ctx.refs에 저장 (다음 단계에서도 참조 가능)
        ctx.refs["artifacts"] = res

        payroll_run_id = self._make_run_id("PR", period)
        ctx.refs["payroll_run_id"] = payroll_run_id
        ctx.history.append({"state": STATE_PAYROLL_CALC, "summary": "급여 산정 조회 실행", "ref": payroll_run_id})

        # ✅ 다음 상태로 이동
        ctx.state = STATE_TAX_CALC

        rows = _to_rows(res.get("result"))
        if rows and isinstance(rows[0], (list, tuple)) and len(rows[0]) >= 4:
            employee_count, total_gross, total_deductions, missing_cnt = rows[0][:4]
            reply = (
                f"✅ 급여 산정(미리보기) 완료\n"
                f"- 대상 인원: {employee_count}명\n"
                f"- 총급여(기본+수당): {_fmt_won(total_gross)}\n"
                f"- 총공제(deductions): {_fmt_won(total_deductions)}\n"
                f"- 누락/오류(급여데이터 없음): {missing_cnt}건\n\n"
                f"다음 단계로 **공제 검증**을 진행할까요?"
            )
        else:
            reply = (
                "급여 산정 결과를 해석할 수 없습니다.\n"
                "- 원인: SQL 결과가 (인원, 총급여, 총공제, 누락건수) 4개 컬럼 1행 형태가 아닐 수 있습니다.\n"
                "- 조치: 실행된 SQL 결과 포맷을 확인해 주세요."
            )

        return {
            "handled": True,
            "reply": reply,
            "state": ctx.state,
            "artifacts": res,
            "suggestions": ["공제 검증 진행", "대상/기간 변경", "취소(시나리오 종료)"],
        }

    # -----------------------------
    # STATE 2) TAX_CALC (공제 검증)
    # -----------------------------
    def _state_tax_calc(self, ctx: ScenarioContext, sql_engine) -> dict:
        ok, missing = self._require(ctx, ["period"])
        if not ok:
            return {
                "handled": True,
                "reply": "먼저 급여 기간(period)이 필요합니다. 예: '2026-01 공제 검증 진행'",
                "state": ctx.state,
                "suggestions": ["이번달 공제 검증", "2026-01 공제 검증"],
            }

        period = ctx.slots["period"]
        payroll_run_id = ctx.refs.get("payroll_run_id")
        if not payroll_run_id:
            ctx.state = STATE_PAYROLL_CALC
            return {
                "handled": True,
                "reply": "공제 검증 전에 급여 산정이 필요합니다. '2026-01 전 직원 급여 산정'처럼 요청해줘.",
                "state": ctx.state,
                "suggestions": ["2026-01 전 직원 급여 산정", "이번달 전 직원 급여 산정"],
            }

        # ✅ 여기서는 스키마에 맞게 '공제 검증' 쿼리로 고정 (0 컬럼 제거)
        # payroll 컬럼: base_salary, bonus, deductions, net_pay
        question = (
            f"{period} payroll 기준으로 공제 검증/요약을 1행으로 조회해줘. "
            f"컬럼 alias는 반드시 아래와 같아야 해:\n"
            f"- employee_count\n"
            f"- total_gross_pay (base_salary + bonus)\n"
            f"- total_deductions\n"
            f"- total_net_pay\n"
            f"- avg_deduction_rate (total_deductions / total_gross_pay)\n"
            f"- zero_deduction_count (deductions=0)\n"
            f"pay_month는 해당 월의 1일(예: DATE '2026-01-01')로 필터링하고 SQL만 출력."
        )

        res = sql_engine.run(question)
        ctx.refs["artifacts"] = res

        tax_run_id = self._make_run_id("TX", period)
        ctx.refs["tax_run_id"] = tax_run_id
        ctx.history.append({"state": STATE_TAX_CALC, "summary": "공제 검증 조회 실행", "ref": tax_run_id})

        ctx.state = STATE_PAYMENT_RUN

        rows = _to_rows(res.get("result"))
        if rows and isinstance(rows[0], (list, tuple)) and len(rows[0]) >= 6:
            employee_count, gross, deductions, net_pay, rate, zero_cnt = rows[0][:6]
            try:
                rate_pct = float(rate) * 100.0
            except Exception:
                rate_pct = rate

            reply = (
                f"✅ 공제 검증 완료\n"
                f"- 대상 인원: {employee_count}명\n"
                f"- 총급여(기본+수당): {_fmt_won(gross)}\n"
                f"- 총공제(deductions): {_fmt_won(deductions)}\n"
                f"- 총실지급(net_pay): {_fmt_won(net_pay)}\n"
                f"- 평균 공제율: {rate_pct:.2f}%\n"
                f"- 공제 0원 인원: {zero_cnt}명\n\n"
                f"※ 현재 DB(payroll)에는 원천세/지방세/4대보험 컬럼이 없어 ‘세금’을 분리 계산할 수 없습니다.\n\n"
                f"다음 단계로 **지급 처리**를 진행할까요?"
            )
        else:
            reply = (
                "공제 검증 결과를 해석할 수 없습니다.\n"
                "- 원인: SQL 결과가 (인원, 총급여, 총공제, 총실지급, 공제율, 공제0인원) 6개 컬럼 1행 형태가 아닐 수 있습니다.\n"
                "- 조치: 실행된 SQL 결과 포맷을 확인해 주세요."
            )

        return {
            "handled": True,
            "reply": reply,
            "state": ctx.state,
            "artifacts": res,
            "suggestions": ["지급 진행", "2026-01-25 지급", "지급일 미정"],
        }

    # -----------------------------
    # STATE 3) PAYMENT_RUN
    # -----------------------------
    def _state_payment_run(self, ctx: ScenarioContext, sql_engine, confirm: Optional[bool]) -> dict:
        ok, missing = self._require(ctx, ["period"])
        if not ok:
            return {
                "handled": True,
                "reply": "지급 처리를 위해 period가 필요합니다.",
                "state": ctx.state,
                "suggestions": ["이번달 지급", "2026-01 지급"],
            }

        period = ctx.slots["period"]
        tax_run_id = ctx.refs.get("tax_run_id")
        if not tax_run_id:
            ctx.state = STATE_TAX_CALC
            return {
                "handled": True,
                "reply": "지급 처리 전에 공제 검증이 필요합니다. '공제 검증 진행'이라고 말해줘.",
                "state": ctx.state,
                "suggestions": ["공제 검증 진행"],
            }

        ok, missing = self._require(ctx, ["pay_date"])
        if not ok:
            return {
                "handled": True,
                "reply": "지급일(pay_date)이 필요합니다. 예: '25일 지급' 또는 '2026-01-25 지급'",
                "state": ctx.state,
                "suggestions": ["25일 지급", "2026-01-25 지급"],
            }

        payment_method = ctx.slots.get("payment_method") or "bank_transfer"
        ctx.slots["payment_method"] = payment_method

        if confirm is None:
            reply = (
                f"지급 실행은 되돌리기 어려울 수 있어요.\n"
                f"- period={period}\n"
                f"- tax_run_id={tax_run_id}\n"
                f"- pay_date={ctx.slots['pay_date']}\n"
                f"- method={payment_method}\n\n"
                f"**지급 실행할까요?** (예/아니오)"
            )
            return {"handled": True, "reply": reply, "state": ctx.state, "suggestions": ["예", "아니오"]}

        if confirm is False:
            return {
                "handled": True,
                "reply": "지급 실행을 취소했습니다. (계속하려면 '예' 또는 '지급 실행'이라고 말해줘)",
                "state": ctx.state,
                "suggestions": ["지급 실행", "지급일 수정", "취소(시나리오 종료)"],
            }

        question = (
            f"{period} 급여 지급 실행을 위한 대상/금액/계좌 오류를 점검하고 "
            f"{ctx.slots['pay_date']} 지급 기준으로 (성공 대상 수, 오류 수, 지급 총액) 1행 요약이 나오도록 조회해줘. "
            f"기준 tax_run_id는 {tax_run_id}야."
        )
        res = sql_engine.run(question)
        ctx.refs["artifacts"] = res

        payment_run_id = self._make_run_id("PM", period)
        ctx.refs["payment_run_id"] = payment_run_id
        ctx.history.append({"state": STATE_PAYMENT_RUN, "summary": "지급 실행(또는 준비) 조회 실행", "ref": payment_run_id})

        ctx.state = STATE_JOURNAL_POST
        reply = (
            f"✅ 지급 처리(준비/실행) 조회를 실행했습니다.\n"
            f"- payment_run_id={payment_run_id}\n\n"
            f"다음 단계는 **전표 생성**입니다.\n"
            f"전표일을 입력해줘. 예: '2026-01-31 전표' 또는 '1/31 전표'"
        )
        return {
            "handled": True,
            "reply": reply,
            "state": ctx.state,
            "artifacts": res,
            "suggestions": ["2026-01-31 전표", "1/31 전표"],
        }

    # -----------------------------
    # STATE 4) JOURNAL_POST
    # -----------------------------
    def _state_journal_post(self, ctx: ScenarioContext, sql_engine, confirm: Optional[bool]) -> dict:
        ok, missing = self._require(ctx, ["period"])
        if not ok:
            return {
                "handled": True,
                "reply": "전표 생성을 위해 period가 필요합니다.",
                "state": ctx.state,
                "suggestions": ["이번달 전표", "2026-01 전표"],
            }

        period = ctx.slots["period"]
        payment_run_id = ctx.refs.get("payment_run_id")
        if not payment_run_id:
            ctx.state = STATE_PAYMENT_RUN
            return {
                "handled": True,
                "reply": "전표 생성 전에 지급 단계가 필요합니다. '지급 실행'부터 진행해줘.",
                "state": ctx.state,
                "suggestions": ["지급 실행"],
            }

        last_text = ctx.slots.get("_last_user_text", "")

        if not ctx.slots.get("journal_date"):
            m = re.search(r"\b(20\d{2})[-./](0?[1-9]|1[0-2])[-./](0?[1-9]|[12]\d|3[01])\b", last_text)
            if m:
                y = int(m.group(1)); mm = int(m.group(2)); dd = int(m.group(3))
                ctx.slots["journal_date"] = f"{y}-{mm:02d}-{dd:02d}"
            else:
                m = re.search(r"\b(0?[1-9]|1[0-2])\s*/\s*(0?[1-9]|[12]\d|3[01])\b", last_text)
                if m:
                    mm = int(m.group(1)); dd = int(m.group(2))
                    y = int(period.split("-")[0])
                    ctx.slots["journal_date"] = f"{y}-{mm:02d}-{dd:02d}"

        if not ctx.slots.get("journal_date"):
            return {
                "handled": True,
                "reply": "전표일(journal_date)이 필요합니다. 예: '2026-01-31 전표' 또는 '1/31 전표'",
                "state": ctx.state,
                "suggestions": ["2026-01-31 전표", "1/31 전표"],
            }

        coa_mapping_version = ctx.slots.get("coa_mapping_version") or "v1"
        ctx.slots["coa_mapping_version"] = coa_mapping_version

        if confirm is None:
            reply = (
                f"전표 생성은 회계에 영향을 줄 수 있어요.\n"
                f"- period={period}\n"
                f"- payment_run_id={payment_run_id}\n"
                f"- journal_date={ctx.slots['journal_date']}\n"
                f"- coa_mapping_version={coa_mapping_version}\n\n"
                f"**전표 생성을 진행할까요?** (예/아니오)"
            )
            return {"handled": True, "reply": reply, "state": ctx.state, "suggestions": ["예", "아니오"]}

        if confirm is False:
            return {
                "handled": True,
                "reply": "전표 생성을 취소했습니다. (계속하려면 '예' 또는 '전표 생성'이라고 말해줘)",
                "state": ctx.state,
                "suggestions": ["전표 생성", "전표일 수정", "취소(시나리오 종료)"],
            }

        question = (
            f"{period} 급여 지급({payment_run_id}) 기준으로 전표 초안을 생성한다고 가정하고, "
            f"{ctx.slots['journal_date']} 일자에 맞춰 계정과목 매핑({coa_mapping_version}) 결과를 "
            f"(차변 합계, 대변 합계, 계정과목별 합계)로 검증/요약 가능하게 조회해줘."
        )
        res = sql_engine.run(question)
        ctx.refs["artifacts"] = res

        journal_run_id = self._make_run_id("JV", period)
        ctx.refs["journal_run_id"] = journal_run_id
        ctx.history.append({"state": STATE_JOURNAL_POST, "summary": "전표 생성(또는 초안) 조회 실행", "ref": journal_run_id})

        ctx.state = STATE_DONE
        reply = (
            f"✅ 전표 단계 조회를 실행했습니다.\n"
            f"- journal_run_id={journal_run_id}\n\n"
            f"전체 프로세스가 완료되었습니다. 요약을 보여드릴까요? (예/아니오)"
        )
        return {
            "handled": True,
            "reply": reply,
            "state": ctx.state,
            "artifacts": res,
            "suggestions": ["예", "아니오", "요약 보여줘"],
        }

    # -----------------------------
    # DONE
    # -----------------------------
    def _state_done(self, ctx: ScenarioContext) -> dict:
        lines = [
            "✅ 급여 E2E 시나리오 완료 요약:",
            f"- period: {ctx.slots.get('period')}",
            f"- scope: {ctx.slots.get('employee_scope')}",
            f"- payroll_run_id: {ctx.refs.get('payroll_run_id')}",
            f"- tax_run_id: {ctx.refs.get('tax_run_id')}",
            f"- payment_run_id: {ctx.refs.get('payment_run_id')}",
            f"- journal_run_id: {ctx.refs.get('journal_run_id')}",
        ]
        # soft clear
        ctx.active_scenario = ""
        ctx.state = ""
        return {"handled": True, "reply": "\n".join(lines), "state": None, "suggestions": ["처음부터 다시", "취소(시나리오 종료)"]}


# -----------------------------
# 4) Orchestrator
# -----------------------------
class ScenarioOrchestrator:
    """
    Routes user input to scenario first, otherwise falls back to raw SQL engine.
    """
    def __init__(self, sql_engine, scenarios: List[PayrollScenario]):
        self.sql_engine = sql_engine
        self.scenarios = scenarios

    def run(self, session_id: str, user_text: str) -> dict:
        for s in self.scenarios:
            out = s.route_and_handle(session_id=session_id, user_text=user_text, sql_engine=self.sql_engine)
            if out.get("handled"):
                return out

        # fallback
        res = self.sql_engine.run(user_text)
        return {
            "handled": True,
            "reply": "요청을 실행했습니다.",
            "state": None,
            "artifacts": res,
            "suggestions": [],
        }
