from pydantic import BaseModel, field_validator
from typing import List, Literal


class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str
    bloom_level: Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


class NotesArtifact(BaseModel):
    artifact_type: Literal["class_notes"]
    topic: str
    subtopic: str
    content: str


class SlidesArtifact(BaseModel):
    artifact_type: Literal["slides"]
    topic: str
    subtopic: str
    slides: List[dict]

    @field_validator("slides")
    @classmethod
    def validate_slides(cls, v):
        for slide in v:
            assert "title" in slide and "bullets" in slide, "Each slide needs title and bullets"
            assert 3 <= len(slide["bullets"]) <= 6, (
                f"Slide '{slide['title']}' must have 3-6 bullets"
            )
        return v


class QuizArtifact(BaseModel):
    artifact_type: Literal["quiz"]
    topic: str
    subtopic: str
    questions: List[QuizQuestion]

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v):
        assert len(v) == 5, f"Quiz must have exactly 5 questions, got {len(v)}"
        levels = {q.bloom_level for q in v}
        required = {"Remember", "Understand", "Apply", "Analyze"}
        covered = required.intersection(levels)
        assert len(covered) == 4, (
            f"Quiz must cover Remember, Understand, Apply, Analyze. Got: {levels}"
        )
        return v


class TakeawaysArtifact(BaseModel):
    artifact_type: Literal["takeaways"]
    topic: str
    subtopic: str
    takeaways: List[str]

    @field_validator("takeaways")
    @classmethod
    def validate_takeaway_count(cls, v):
        assert 3 <= len(v) <= 5, f"Takeaways must be 3-5 items, got {len(v)}"
        return v