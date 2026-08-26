from openai import OpenAI
from google import genai

import os
from dotenv import load_dotenv

load_dotenv()
## open AI API 키
open_api_key = os.getenv("OPENAI_API_KEY")
client_0 = OpenAI(api_key=open_api_key)

## gemini api 키
gemini_api_key = os.getenv("GEMINI_API_KEY")
client_1 = genai.Client()

response_open_api = client_0.responses.create(
    model="gpt-5.6",
    input="Write a one-sentence bedtime story about a unicorn.",
)


        
response_gemini = client_1.models.generate_content(
    model="gemini-3.6-flash", 
    contents="Explain how AI works in a few words"
)

print(response_open_api.text) 

print(response_gemini.text)