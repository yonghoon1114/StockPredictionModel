from dotenv import load_dotenv
import os
 
load_dotenv()

DART_API_KEY = os.getenv("DART_API_KEY")
GEMINI_API_KEY = os.getenv("Gemini_API_KEY")