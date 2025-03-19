import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY not set!")
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

try:
    test_response = gemini_model.generate_content("Hello, how are my tomato plants?")
    print("Test response:", test_response.text)
except Exception as e:
    print("Error calling Gemini API:", e)
