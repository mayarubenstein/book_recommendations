import json
import os
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class BookEvaluation(BaseModel):
    title: str
    hard_constraint_check: str = Field(description="Check avoid_list and negative free_text. Yes/No and explain.")
    soft_constraint_check: str = Field(description="Evaluate match with genres, moods, and tone.")
    is_approved: bool = Field(description="True ONLY IF hard constraints are respected and soft constraints match.")
    rejection_reason: Optional[str]

class VerificationResult(BaseModel):
    evaluations: List[BookEvaluation]

def verify_books(user_json: dict, candidate_books: List[dict]):
    clean_profile = {k: v for k, v in user_json.items() if v}
    system_prompt = """
    You are a Book Recommendation Auditor. Evaluate candidate books against the user's JSON profile.
    1. HARD CONSTRAINTS: Reject immediately if the book contains items in 'avoid_list' or negative 'free_text'.
    2. SOFT CONSTRAINTS: Must align with genres/moods.
    3. Retrieve internal knowledge about the book's plot based on Title/Author to verify constraints.
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"PROFILE:\n{json.dumps(clean_profile)}\n\nCANDIDATES:\n{json.dumps(candidate_books)}"}
        ],
        response_format=VerificationResult,
    )
    return response.choices[0].message.parsed