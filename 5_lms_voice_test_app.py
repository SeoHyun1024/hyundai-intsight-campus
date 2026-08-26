# =======================================
# 물류회사 AI 데모 (Streamlit + Gemini)
#
# 실행 : streamlit run 5_lms_voice_test_app.py
# ----------------------------------------

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


st.set_page_config(
    page_title="물류 AI 음성 상담",
    page_icon=":material/local_shipping:",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
GEMINI_MODEL = "gemini-3.6-flash"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(BASE_DIR / "data" / "logistics_inventory.csv"),
        pd.read_csv(BASE_DIR / "data" / "logistics_deliveries.csv"),
        pd.read_csv(BASE_DIR / "data" / "logistics_accidents.csv"),
    )


@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


inventory, deliveries, accidents = load_data()
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = get_gemini_client(api_key) if api_key else None


def data_context() -> str:
    """Gemini가 답변에 사용할 물류 데이터 원문을 만든다."""
    return "\n\n".join(
        [
            "[재고 데이터]\n" + inventory.to_csv(index=False),
            "[배송 데이터]\n" + deliveries.to_csv(index=False),
            "[사고 데이터]\n" + accidents.to_csv(index=False),
        ]
    )


def generate_response(prompt: str) -> str:
    if client is None:
        return "GEMINI_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 입력해 주세요."

    instruction = (
        "당신은 물류회사 상담 AI입니다. 아래 물류 데이터만 근거로 답변하세요. "
        "데이터에 없는 수치나 사실은 추측하지 말고, 모른다고 말하세요. "
        "한국어로 간결하게 답하고, 사용자의 질문이 재고 부족·배송 지연·사고 보고서와 "
        "관련되면 핵심 수치와 근거를 함께 제시하세요.\n\n"
        + data_context()
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0.1,
        ),
    )
    return response.text or "답변을 생성하지 못했습니다. 질문을 다시 입력해 주세요."


def transcribe_audio(audio_bytes: bytes, mime_type: str) -> str:
    """업로드된 음성 파일을 Gemini로 한국어 텍스트로 변환한다."""
    if client is None:
        return ""

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type or "audio/wav",
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    audio_part,
                    types.Part.from_text(
                        text="이 음성의 내용을 한국어 질문 한 문장으로 정확하게 받아 적어 주세요."
                    ),
                ],
            )
        ],
        config=types.GenerateContentConfig(temperature=0),
    )
    return (response.text or "").strip()


def add_exchange(
    question: str = "", audio_bytes: bytes | None = None, mime_type: str = ""
) -> None:
    question = question.strip()
    if audio_bytes is not None:
        with st.status(
            "음성 질문을 분석하고 있습니다...", expanded=True
        ) as transcription_status:
            st.write("음성 파일을 Gemini로 전송하는 중입니다.")
            question = transcribe_audio(
                audio_bytes, mime_type or "audio/wav"
            )
            if question:
                transcription_status.update(
                    label="음성 인식이 완료되었습니다.", state="complete"
                )
            else:
                transcription_status.update(
                    label="음성 인식에 실패했습니다.", state="error"
                )
        if not question:
            st.error("음성을 인식하지 못했습니다.", icon=":material/error:")
            return

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status(
            "답변을 준비하고 있습니다...", expanded=True
        ) as response_status:
            st.write("물류 CSV 데이터를 검색하는 중입니다.")
            answer = generate_response(question)
            response_status.update(
                label="답변 준비가 완료되었습니다.", state="complete"
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if "messages" not in st.session_state:
    st.session_state.messages = []
if "sample_audio_path" not in st.session_state:
    st.session_state.sample_audio_path = None
if "pending_request" not in st.session_state:
    st.session_state.pending_request = None


st.title("물류 AI 음성 상담")
st.caption("Gemini가 음성을 질문으로 변환하고, 물류 CSV 데이터를 근거로 답변합니다.")

with st.sidebar:
    st.header("음성 질문")
    uploaded_audio = st.file_uploader(
        "음성 파일 업로드",
        type=["mp3", "wav", "m4a", "webm"],
        help="음성 파일을 올리면 Gemini가 질문을 인식합니다.",
    )
    if uploaded_audio is not None:
        st.audio(uploaded_audio)
        if st.button("음성 질문 보내기", type="primary", width="stretch"):
            st.session_state.pending_request = {
                "audio_bytes": uploaded_audio.getvalue(),
                "mime_type": uploaded_audio.type or "audio/wav",
            }

    st.subheader("데모 음성")
    sample_paths = sorted((BASE_DIR / "data").glob("*.mp3"))
    sample_labels = {path.name: path for path in sample_paths}
    selected_sample = st.selectbox(
        "샘플 선택",
        options=["선택 안 함", *sample_labels.keys()],
        label_visibility="collapsed",
    )
    if selected_sample != "선택 안 함":
        sample_path = sample_labels[selected_sample]
        st.audio(str(sample_path))
        if st.button("샘플 질문 보내기", width="stretch"):
            with open(sample_path, "rb") as audio_stream:
                st.session_state.pending_request = {
                    "audio_bytes": audio_stream.read(),
                    "mime_type": "audio/mpeg",
                }

    st.divider()
    st.subheader("데이터 요약")
    st.metric("물류센터", int(inventory["center"].nunique()))
    st.metric("재고 항목", len(inventory))
    delayed = deliveries[
        (deliveries["week"] == "이번주") & (deliveries["status"] == "지연")
    ]
    st.metric("이번 주 지연", f"{int(delayed['count'].sum())}건")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


chat_input = st.chat_input(
    "질문을 입력하거나 음성 버튼을 눌러 보세요",
    accept_audio=True,
    audio_sample_rate=16000,
)
if chat_input:
    if chat_input.audio:
        st.session_state.pending_request = {
            "audio_bytes": chat_input.audio.getvalue(),
            "mime_type": chat_input.audio.type or "audio/wav",
        }
    else:
        st.session_state.pending_request = {"question": chat_input.text or ""}


pending_request = st.session_state.pending_request
st.session_state.pending_request = None
if pending_request:
    add_exchange(
        pending_request.get("question", ""),
        pending_request.get("audio_bytes"),
        pending_request.get("mime_type", ""),
    )