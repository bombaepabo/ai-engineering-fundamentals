# app/services/gemini_service.py
import os
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import settings

# Define the structured output schema we expect from Gemini
class GeminiAnalysisResult(BaseModel):
    summary: str
    suggested_reply: str
    sentiment: str  # positive | neutral | negative


class GeminiService:
    """
    Wrapper around the Google GenAI SDK to interact with the Gemini model.
    """
    def __init__(self):
        self.client = None
        
    def _get_client(self):
        if self.client is None:
            # Initialize GenAI Client using the configured API Key
            self.client = genai.Client(api_key=settings.gemini_api_key)
        return self.client

    def analyze_ticket(self, subject: str, message: str, context: str = "") -> GeminiAnalysisResult:
        """
        Calls Gemini to summarize the ticket, detect sentiment, and draft a response.
        Uses the provided context to ground the response.
        """
        prompt_path = "app/prompts/analysis_prompt.txt"
        
        # Load the prompt template
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
        else:
            template = (
                "Analyze this ticket:\n"
                "Subject: {subject}\n"
                "Message: {message}\n"
                "Context: {context}\n"
                "Provide a summary, suggested reply, and sentiment."
            )
        # Pass context into the formatter
        prompt = template.format(subject=subject, message=message, context=context)
        client = self._get_client()
        try:
            # Query Gemini with structured output
            response = client.models.generate_content(
                model=settings.gemini_chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiAnalysisResult,
                    temperature=0.2,
                ),
            )
            result = GeminiAnalysisResult.model_validate_json(response.text)
            return result
        except Exception as e:
            print(f"Gemini API call failed: {e}")
            return GeminiAnalysisResult(
                summary="[Failed to generate summary]",
                suggested_reply="Dear customer, we have received your request and are looking into it.",
                sentiment="neutral"
            )
gemini_service = GeminiService()