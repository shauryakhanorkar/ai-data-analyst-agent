import os

from dotenv import load_dotenv
from google import genai


# Find the main project folder and load its .env file.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def create_gemini_client():
    """Create a Gemini client using the secret API key."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Check your .env file in the project root."
        )

    return genai.Client(api_key=api_key)


def ask_gemini(prompt):
    """Send a prompt to Gemini and return its text response."""

    client = create_gemini_client()

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text


if __name__ == "__main__":
    question = "Explain data analysis in three simple sentences."

    try:
        answer = ask_gemini(question)
        print("\nGemini response:")
        print(answer)

    except Exception as error:
        print(f"\nGemini request failed: {error}")
        