import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class BookEvaluation(BaseModel):
    book_id: str | None = None
    title: str
    authors: str | None = None
    hard_constraint_check: str = Field(description="Check avoid_list and negative free_text. Yes/No and explain.")
    soft_constraint_check: str = Field(description="Evaluate match with genres, moods, and tone.")
    is_approved: bool = Field(description="True ONLY IF hard constraints are respected and soft constraints match.")
    rejection_reason: str | None = None

class VerificationResult(BaseModel):
    evaluations: list[BookEvaluation]


def verify_books(
    user_json: dict[str, Any],
    candidate_books: list[dict[str, Any]],
    client: OpenAI | None = None,
) -> VerificationResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if client is None:
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to the backend .env file."
            )
        client = OpenAI(api_key=api_key, timeout=90.0, max_retries=0)

    clean_profile = {k: v for k, v in user_json.items() if v}
    system_prompt = """
    You are a Book Recommendation Auditor. Evaluate candidate books against the user's JSON profile.
    1. HARD CONSTRAINTS: Reject immediately if the book contains items in 'avoid_list' or negative 'free_text'.
    2. SOFT CONSTRAINTS: Must align with genres/moods.
    3. Use the supplied candidate metadata and, when needed, your knowledge of the book's plot.
    4. Return exactly one evaluation per candidate and preserve each candidate's book_id and title.
    5. Treat different IDs for the same normalized title/author as one book; do not approve duplicates.
     6. Approve only books that are clearly English-language books. Reject books in Polish,
         Arabic, or any other non-English language, even when the title is transliterated.
    """
    started_at = time.perf_counter()
    print(f"[verify] sending {len(candidate_books)} candidates to OpenAI")
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"PROFILE:\n{json.dumps(clean_profile)}\n\nCANDIDATES:\n{json.dumps(candidate_books)}"}
        ],
        response_format=VerificationResult,
    )
    print(f"[verify] received response in {time.perf_counter() - started_at:.1f}s")
    return response.choices[0].message.parsed