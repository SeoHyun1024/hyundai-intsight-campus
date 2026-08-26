#=======================================
#  물류회사 AI 데모 (Streamlit 버전)
#
# 실행 : streamlit run 4_lms_test_app.py
#----------------------------------------

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
inventory = pd.read_csv(BASE_DIR / "data" / "logistics_inventory.csv")
deliveries = pd.read_csv(BASE_DIR / "data" / "logistics_deliveries.csv")
accidents = pd.read_csv(BASE_DIR / "data" / "logistics_accidents.csv")


# --------------------------------------------------
# 실제 함수
# --------------------------------------------------
def get_low_stock(center: str) -> str:
    """물류센터명(예: 강남센터)을 받아 재고 20개 이하인 재고 부족 상품 목록을 반환한다. 재고 부족 질문에 사용."""
    row = inventory[(inventory["center"] == center) & (inventory["stock"] <= 20)]
    if row.empty:
        return f"'{center}'에는 재고 부족 상품이 없습니다."
    names = ", ".join(row["product_name"].tolist())
    return f"현재 재고가 20개 이하인 상품은 {names}입니다."


def get_delivery_delay_count(region: str) -> str:
    """지역명(예: 부산)을 받아 이번 주 배송 지연 건수를 반환한다. 배송 지연 질문에 사용."""
    row = deliveries[
        (deliveries["region"] == region)
        & (deliveries["week"] == "이번주")
        & (deliveries["status"] == "지연")
    ]
    if row.empty:
        return f"'{region}' 지역의 이번 주 배송 지연 데이터가 없습니다."
    count = int(row["count"].sum())
    return f"이번 주 {region} 지역 배송 지연은 {count}건입니다."


def generate_accident_report(keyword: str) -> str:
    """사고 유형 키워드(예: 파손, 분실, 오배송)를 받아 해당 사고들을 정리한 보고서를 작성한다. 사고 보고서 작성 요청에 사용."""
    row = accidents[accidents["type"].str.contains(keyword, na=False)]
    if row.empty:
        return f"'{keyword}' 관련 사고 이력이 없습니다."

    total_qty = int(row["quantity"].sum())
    lines = [f"[{keyword} 사고 보고서] 총 {len(row)}건, 수량 {total_qty}개"]
    for _, r in row.iterrows():
        lines.append(
            f"- {r['date']} {r['product_name']} {int(r['quantity'])}개: {r['description']}"
        )
    report = "\n".join(lines)
    return f"물류 사고 보고서를 작성했습니다.\n{report}"


# --------------------------------------------------
# 함수 등록
# --------------------------------------------------
FUNCTIONS = {
    "get_low_stock": get_low_stock,
    "get_delivery_delay_count": get_delivery_delay_count,
    "generate_accident_report": generate_accident_report,
}


# --------------------------------------------------
# Tool 스키마 자동 생성
# --------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "name": name,
        "description": func.__doc__ or "",
        "parameters": {
            "type": "object",
            "properties": {
                list(func.__code__.co_varnames[: func.__code__.co_argcount])[0]: {
                    "type": "string"
                }
            },
            "required": [
                list(func.__code__.co_varnames[: func.__code__.co_argcount])[0]
            ],
        },
    }
    for name, func in FUNCTIONS.items()
]


# ============================
# OpenAI API를 사용
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
OPENAI_MODEL = "gpt-4.1-mini"


# --------------------------------------------------
# 질문 함수
# --------------------------------------------------
def ask(question):
    """질문을 받아 모델이 알맞은 도구를 골라 실행하고(자동), 최종 답변을 돌려준다."""
    messages = [{"role": "user", "content": question}]

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=messages,
        tools=TOOLS,
        temperature=0,
    )

    function_calls = [item for item in response.output if item.type == "function_call"]

    if not function_calls:
        return response.output_text

    messages += response.output

    for call in function_calls:
        result = FUNCTIONS[call.name](**json.loads(call.arguments))

        messages.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result,
            }
        )

    final = client.responses.create(
        model=OPENAI_MODEL,
        input=messages,
    )

    return final.output_text


# --------------------------------------------------
# 화면 (Streamlit)
# --------------------------------------------------
st.set_page_config(page_title="물류 AI 데모", page_icon="🚚")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #020817 0%, #0f172a 100%);
            color: #f8fafc;
        }
        .stApp * {
            color: #f8fafc;
        }
        .app-header {
            background: linear-gradient(135deg, #0b1f3a 0%, #1d4ed8 52%, #2563eb 100%);
            border-radius: 22px;
            padding: 1.6rem 1.7rem 1.3rem 1.7rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 30px rgba(30, 64, 175, 0.18);
        }
        .app-title {
            color: #f8fbff;
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.05em;
        }
        .app-subtitle {
            color: rgba(255,255,255,0.85);
            font-size: 0.98rem;
            margin-top: 0.45rem;
            line-height: 1.5;
        }
        .section-tag {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: #dbeafe;
            color: #0f172a;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.55rem;
        }
        .feature-card {
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: rgba(15, 23, 42, 0.72);
            margin-bottom: 0.75rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.1);
        }
        .feature-card strong {
            display: block;
            margin-bottom: 0.25rem;
            color: #f8fafc;
            font-size: 1rem;
        }
        .feature-card span {
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .chat-shell {
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 18px;
            padding: 0.85rem 0.9rem 0.8rem 0.9rem;
            background: rgba(15, 23, 42, 0.72);
            min-height: 420px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .stChatMessage {
            padding: 0.2rem 0.2rem 0.5rem 0.2rem;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.25rem 0.4rem;
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid #60a5fa;
            background: linear-gradient(180deg, #1e3a8a 0%, #1d4ed8 100%);
            color: #f8fbff;
            font-weight: 800;
            box-shadow: 0 3px 12px rgba(96, 165, 250, 0.25);
            transition: all 0.2s ease;
            min-height: 2.8rem;
        }
        .stButton > button:hover {
            border-color: #93c5fd;
            background: linear-gradient(180deg, #2563eb 0%, #3b82f6 100%);
            box-shadow: 0 8px 18px rgba(96, 165, 250, 0.35);
            color: #ffffff;
            transform: translateY(-1px);
        }
        .stButton > button:focus {
            outline: 3px solid rgba(147, 197, 253, 0.75);
            outline-offset: 2px;
        }
        [data-testid="stForm"] {
            margin-top: 1rem;
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 18px;
            padding: 0.9rem 1rem 0.2rem 1rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.14);
        }
        div[data-baseweb="base-input"] {
            border-radius: 12px;
            border-color: #93c5fd;
        }
        .stTextInput input {
            border-radius: 12px;
            border: 1px solid #60a5fa;
            background: rgba(15, 23, 42, 0.7);
            color: #f8fafc;
            font-size: 1rem;
            padding: 0.75rem 0.9rem;
        }
        .stTextInput input:focus {
            border-color: #93c5fd;
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.25);
        }
        .stMetric {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.18);
        }
        .stMetric > div {
            color: #f8fafc !important;
        }
        .stSidebar {
            background: rgba(2, 8, 23, 0.95);
        }
        .stSidebar * {
            color: #f8fafc !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">🚚 물류 AI 데모</div>
        <div class="app-subtitle">재고 부족, 배송 지연, 사고 보고서까지 한 번에 확인할 수 있습니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = [
    "인천센터 재고 부족 상품 알려줘",
    "부산 배송 지연 건수는?",
    "파손 사고 내용을 보고서로 만들어줘",
    "강남센터 재고가 얼마 남았어?",
]

with st.sidebar:
    st.markdown("### 🧭 기능 메뉴")
    st.markdown(
        """
        <div class="feature-card">
            <strong>재고 조회</strong>
            <span>센터별 재고 부족 상품을 빠르게 확인합니다.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-card">
            <strong>배송 지연</strong>
            <span>지역별 이번 주 지연 건수를 조회합니다.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-card">
            <strong>사고 보고서</strong>
            <span>파손·분실·오배송 등 사고 내용을 정리합니다.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📊 실시간 요약")
    st.metric("물류센터 수", inventory["center"].nunique())
    st.metric("재고 데이터 건수", len(inventory))
    st.metric("지연 건수 합계", int(deliveries[(deliveries["week"] == "이번주") & (deliveries["status"] == "지연")]["count"].sum()))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_box" not in st.session_state:
    st.session_state.input_box = ""


def _fill_example(question):
    st.session_state.input_box = question


st.markdown('<div class="section-tag">QUESTION</div>', unsafe_allow_html=True)
st.subheader("💬 질문 예시")
example_cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, q in zip(example_cols, EXAMPLE_QUESTIONS):
    col.button(q, use_container_width=True, on_click=_fill_example, args=(q,))

chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)


def _submit():
    question = st.session_state.input_box.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    answer = ask(question)
    st.session_state.messages.append({"role": "assistant", "content": answer})


with st.form(key="question_form"):
    st.text_input("질문을 입력하세요.", key="input_box", placeholder="예: 인천센터 재고 부족 상품 알려줘")
    submitted = st.form_submit_button("전송", use_container_width=True)

    if submitted:
        _submit()
