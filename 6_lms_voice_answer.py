import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


st.set_page_config(
	page_title="물류 AI 음성 답변",
	page_icon=":material/record_voice_over:",
	layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
OPENAI_MODEL = "gpt-4.1-mini"
TTS_MODEL = "gpt-4o-mini-tts"


@st.cache_data
def load_data():
	return (
		pd.read_csv(BASE_DIR / "data" / "logistics_inventory.csv"),
		pd.read_csv(BASE_DIR / "data" / "logistics_deliveries.csv"),
		pd.read_csv(BASE_DIR / "data" / "logistics_accidents.csv"),
	)


@st.cache_resource
def get_client(api_key: str):
	return OpenAI(api_key=api_key)


inventory, deliveries, accidents = load_data()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = get_client(api_key) if api_key else None


def get_low_stock(center: str) -> str:
	"""물류센터명(예: 강남센터)의 재고 20개 이하 상품을 조회한다."""
	rows = inventory[(inventory["center"] == center) & (inventory["stock"] <= 20)]
	if rows.empty:
		return f"'{center}'에는 재고 부족 상품이 없습니다."
	return "현재 재고가 20개 이하인 상품은 " + ", ".join(rows["product_name"].tolist()) + "입니다."


def get_delivery_delay_count(region: str) -> str:
	"""지역명(예: 부산)의 이번 주 배송 지연 건수를 조회한다."""
	rows = deliveries[
		(deliveries["region"] == region)
		& (deliveries["week"] == "이번주")
		& (deliveries["status"] == "지연")
	]
	if rows.empty:
		return f"'{region}' 지역의 이번 주 배송 지연 데이터가 없습니다."
	return f"이번 주 {region} 지역 배송 지연은 {int(rows['count'].sum())}건입니다."


def generate_accident_report(keyword: str) -> str:
	"""사고 유형 키워드(예: 파손, 분실, 오배송)의 사고 보고서를 작성한다."""
	rows = accidents[accidents["type"].str.contains(keyword, na=False)]
	if rows.empty:
		return f"'{keyword}' 관련 사고 이력이 없습니다."
	lines = [f"[{keyword} 사고 보고서] 총 {len(rows)}건, 수량 {int(rows['quantity'].sum())}개"]
	for _, row in rows.iterrows():
		lines.append(
			f"- {row['date']} {row['product_name']} {int(row['quantity'])}개: {row['description']}"
		)
	return "물류 사고 보고서를 작성했습니다.\n" + "\n".join(lines)


FUNCTIONS = {
	"get_low_stock": get_low_stock,
	"get_delivery_delay_count": get_delivery_delay_count,
	"generate_accident_report": generate_accident_report,
}

TOOLS = [
	{
		"type": "function",
		"name": name,
		"description": function.__doc__ or "",
		"parameters": {
			"type": "object",
			"properties": {function.__code__.co_varnames[0]: {"type": "string"}},
			"required": [function.__code__.co_varnames[0]],
		},
	}
	for name, function in FUNCTIONS.items()
]


def ask(question: str) -> str:
	if client is None:
		return "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요."

	messages = [{"role": "user", "content": question}]
	response = client.responses.create(
		model=OPENAI_MODEL,
		input=messages,
		tools=TOOLS,
		temperature=0,
	)
	calls = [item for item in response.output if item.type == "function_call"]
	if not calls:
		return response.output_text

	messages += response.output
	for call in calls:
		result = FUNCTIONS[call.name](**json.loads(call.arguments))
		messages.append(
			{"type": "function_call_output", "call_id": call.call_id, "output": result}
		)
	final_response = client.responses.create(model=OPENAI_MODEL, input=messages)
	return final_response.output_text


def create_audio(text: str) -> bytes:
	if client is None:
		return b""
	response = client.audio.speech.create(
		model=TTS_MODEL,
		voice="alloy",
		input=text,
		response_format="mp3",
	)
	return response.content


def process_question(question: str) -> None:
	question = question.strip()
	if not question:
		return

	st.session_state.messages.append({"role": "user", "content": question})
	with st.chat_message("user"):
		st.markdown(question)

	with st.chat_message("assistant"):
		with st.status("답변을 준비하고 있습니다...", expanded=True) as status:
			st.write("물류 데이터를 확인하는 중입니다.")
			answer = ask(question)
			status.update(label="텍스트 답변이 완성되었습니다.", state="complete")
			st.write("답변을 음성으로 변환하는 중입니다.")
			audio = create_audio(answer)
			status.update(label="음성 답변이 준비되었습니다.", state="complete")
		st.markdown(answer)
		if audio:
			st.audio(audio, format="audio/mp3", autoplay=True)

	st.session_state.messages.append(
		{"role": "assistant", "content": answer, "audio": audio}
	)


if "messages" not in st.session_state:
	st.session_state.messages = []

st.title("물류 AI 음성 답변")
st.caption("OpenAI가 물류 데이터를 분석해 답변하고 음성으로 읽어드립니다.")

with st.sidebar:
	st.subheader("질문 예시")
	examples = [
		"인천센터 재고 부족 상품 알려줘",
		"부산 배송 지연 건수는?",
		"파손 사고 내용을 보고서로 만들어줘",
	]
	for index, example in enumerate(examples):
		if st.button(example, key=f"example_{index}", width="stretch"):
			st.session_state.pending_question = example

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
		if message["role"] == "assistant" and message.get("audio"):
			st.audio(message["audio"], format="audio/mp3")


question = st.chat_input("질문을 입력하세요")
pending_question = st.session_state.pop("pending_question", None)
if question or pending_question:
	process_question(question or pending_question)
