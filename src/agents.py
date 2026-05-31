import os
import re
import json
from groq import Groq
from typing import List
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL = "llama-3.1-8b-instant"


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 1500) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def safe_json(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', match.group())
        try:
            return json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse LLM output as JSON.\nRaw output:\n{text[:500]}")


def build_user_prompt(subtopic: str, context_chunks: List[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    return f"Topic: Python Fundamentals\nSubtopic: {subtopic}\n\nSource material:\n{context}"


NOTES_SYSTEM = """You are an educational content writer for Indian engineering colleges.
Generate concise class notes for faculty delivery. Keep the total content under 600 words.
Cover key concepts, one or two short code examples, and main points only.
The content field must be a single plain string — no newlines, no special characters, spaces only.
Respond ONLY with this JSON object:
{"artifact_type": "class_notes", "topic": "...", "subtopic": "...", "content": "..."}
No markdown fences, no preamble, no newlines inside any string value."""

SLIDES_SYSTEM = """You are a presentation designer for engineering education.
Generate exactly 3 slides. Each slide has a title and exactly 4 to 5 concise bullet points.
No newlines or special characters inside any string values.
Respond ONLY with this JSON object:
{"artifact_type": "slides", "topic": "...", "subtopic": "...", "slides": [{"title": "...", "bullets": ["...", ...]}, ...]}
No markdown fences, no preamble, no newlines inside any string value."""

QUIZ_SYSTEM = """You are a quiz designer for engineering students using Bloom's taxonomy.
Generate EXACTLY 5 MCQ questions:
- Question 1: bloom_level = "Remember"
- Question 2: bloom_level = "Understand"
- Question 3: bloom_level = "Apply"
- Question 4: bloom_level = "Analyze"
- Question 5: bloom_level = "Analyze"
Each question: question (string), options (list of exactly 4 short strings), answer (must be the FULL TEXT of the correct option, not a letter like "a" or "c"), bloom_level.
No newlines or special characters inside any string values.
Respond ONLY with this JSON object:
{"artifact_type": "quiz", "topic": "...", "subtopic": "...", "questions": [...]}
No markdown fences, no preamble, no newlines inside any string value."""

TAKEAWAYS_SYSTEM = """You are a learning outcomes specialist.
Generate exactly 4 key takeaways. Each is one short sentence under 20 words.
No newlines or special characters inside any string values.
Respond ONLY with this JSON object:
{"artifact_type": "takeaways", "topic": "...", "subtopic": "...", "takeaways": ["...", "...", "...", "..."]}
No markdown fences, no preamble, no newlines inside any string value."""


def generate_notes(subtopic: str, chunks: List[dict]) -> dict:
    return safe_json(call_llm(NOTES_SYSTEM, build_user_prompt(subtopic, chunks), max_tokens=1000))

def generate_slides(subtopic: str, chunks: List[dict]) -> dict:
    return safe_json(call_llm(SLIDES_SYSTEM, build_user_prompt(subtopic, chunks), max_tokens=800))

def generate_quiz(subtopic: str, chunks: List[dict]) -> dict:
    return safe_json(call_llm(QUIZ_SYSTEM, build_user_prompt(subtopic, chunks), max_tokens=1000))

def generate_takeaways(subtopic: str, chunks: List[dict]) -> dict:
    return safe_json(call_llm(TAKEAWAYS_SYSTEM, build_user_prompt(subtopic, chunks), max_tokens=400))


ARTIFACT_GENERATORS = {
    "notes": generate_notes,
    "slides": generate_slides,
    "quiz": generate_quiz,
    "takeaways": generate_takeaways,
}