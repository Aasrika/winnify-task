import re
from typing import List
from pydantic import ValidationError
from schemas import NotesArtifact, SlidesArtifact, QuizArtifact, TakeawaysArtifact
from models import NLI_MODEL, ENTAILMENT_IDX


ARTIFACT_SCHEMAS = {
    "notes": NotesArtifact,
    "slides": SlidesArtifact,
    "quiz": QuizArtifact,
    "takeaways": TakeawaysArtifact,
}


def schema_validate(artifact_type: str, data: dict) -> tuple[bool, str]:
    schema = ARTIFACT_SCHEMAS[artifact_type]
    try:
        schema(**data)
        return True, "OK"
    except (ValidationError, AssertionError, Exception) as e:
        return False, str(e)


def extract_claims(data: dict, artifact_type: str) -> List[str]:
    if artifact_type == "notes":
        text = data.get("content", "")
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 30][:10]
    elif artifact_type == "slides":
        claims = []
        for slide in data.get("slides", []):
            claims.extend(slide.get("bullets", []))
        return claims
    elif artifact_type == "quiz":
        return [q["question"] for q in data.get("questions", [])]
    elif artifact_type == "takeaways":
        return data.get("takeaways", [])
    return []


def nli_grounding_validate(
    data: dict,
    artifact_type: str,
    store,
    source_chunk_ids: List[str],
    entailment_threshold: float = 0.3
) -> tuple[bool, float, List[dict]]:
    claims = extract_claims(data, artifact_type)
    if not claims:
        return True, 1.0, []

    source_chunks = store.get_chunks_by_ids(source_chunk_ids)
    if not source_chunks:
        return False, 0.0, []

    premises = [c["text"] for c in source_chunks]

    claim_results = []
    for claim in claims:
        pairs = [(premise, claim) for premise in premises]
        scores = NLI_MODEL.predict(pairs)
        entailment_scores = [float(s[ENTAILMENT_IDX]) for s in scores]
        best_entailment = max(entailment_scores)
        claim_results.append({
            "claim": claim[:120],
            "best_entailment_score": round(best_entailment, 4),
            "grounded": best_entailment >= entailment_threshold
        })

    grounded_count = sum(1 for r in claim_results if r["grounded"])
    coverage_ratio = grounded_count / len(claim_results)
    passed = coverage_ratio >= 0.6

    return passed, round(coverage_ratio, 4), claim_results


def critic_validate(
    artifact_type: str,
    data: dict,
    store,
    source_chunk_ids: List[str]
) -> dict:
    schema_ok, schema_msg = schema_validate(artifact_type, data)
    nli_ok, coverage_ratio, claim_details = nli_grounding_validate(
        data, artifact_type, store, source_chunk_ids
    )

    return {
        "schema_valid": schema_ok,
        "schema_message": schema_msg,
        "nli_coverage_ratio": coverage_ratio,
        "nli_grounding_ok": nli_ok,
        "claim_details": claim_details,
        "passed": schema_ok and nli_ok
    }