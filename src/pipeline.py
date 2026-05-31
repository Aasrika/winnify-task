from ingestion import load_text, recursive_chunk, CS50_NOTES_URLS
from vector_store import VectorStore
from agents import ARTIFACT_GENERATORS
from validator import critic_validate, ARTIFACT_SCHEMAS
import json
import os


def run_pipeline(source: str, subtopic: str) -> dict:
    print(f"\n{'='*60}")
    print(f"WinTeach Pipeline — Subtopic: {subtopic}")
    print(f"{'='*60}")

    print("\n[1/4] Ingesting and chunking source material...")
    raw_text = load_text(source)
    chunks = recursive_chunk(raw_text)
    print(f"      {len(chunks)} chunks created")

    print("\n[2/4] Building FAISS vector store...")
    store = VectorStore(chunks)

    print("\n[3/4] Routing and generating artifacts...")
    context_chunks = store.retrieve(subtopic, k=4)
    raw_results = {}
    for artifact_type, generator in ARTIFACT_GENERATORS.items():
        print(f"  Generating {artifact_type}...")
        raw_results[artifact_type] = {
            "data": generator(subtopic, context_chunks),
            "source_chunk_ids": [c["chunk_id"] for c in context_chunks]
        }

    print("\n[4/4] Running critic validation (Schema + NLI Grounding)...")
    final_output = {}
    for artifact_type, result in raw_results.items():
        data = result["data"]
        source_ids = result["source_chunk_ids"]

        validation = critic_validate(artifact_type, data, store, source_ids)

        if not validation["passed"]:
            print(f"  [{artifact_type}] First attempt failed — retrying once with k=5...")
            richer_chunks = store.retrieve(subtopic, k=5)
            data = ARTIFACT_GENERATORS[artifact_type](subtopic, richer_chunks)
            source_ids = [c["chunk_id"] for c in richer_chunks]
            validation = critic_validate(artifact_type, data, store, source_ids)

        status = "READY FOR REVIEW" if validation["passed"] else "FLAGGED"
        print(
            f"  [{artifact_type}] "
            f"Schema: {'✓' if validation['schema_valid'] else '✗'} | "
            f"NLI Coverage: {validation['nli_coverage_ratio']:.0%} | "
            f"Status: {status}"
        )

        final_output[artifact_type] = {
            "artifact": data,
            "source_chunk_ids": source_ids,
            "validation": validation,
            "status": status
        }

    return final_output


def save_output(output: dict, subtopic: str):
    os.makedirs("../outputs", exist_ok=True)
    safe_name = subtopic.lower().replace(" ", "_").replace("/", "_")
    path = f"../outputs/{safe_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {path}")


def print_preview(output: dict):
    print("\n" + "="*60)
    print("FINAL OUTPUT PREVIEW")
    print("="*60)
    for artifact_type, result in output.items():
        print(f"\n--- {artifact_type.upper()} [{result['status']}] ---")
        artifact = result["artifact"]
        if artifact_type == "notes":
            print(artifact.get("content", "")[:300] + "...")
        elif artifact_type == "slides":
            for slide in artifact.get("slides", []):
                print(f"  Slide: {slide['title']}")
                for b in slide["bullets"]:
                    print(f"    • {b}")
                print()
        elif artifact_type == "quiz":
            for q in artifact.get("questions", []):
                print(f"  [{q['bloom_level']}] {q['question']}")
        elif artifact_type == "takeaways":
            for t in artifact.get("takeaways", []):
                print(f"  • {t}")


def pick_source() -> tuple[str, str]:
    topic_names = list(CS50_NOTES_URLS.keys())

    print("\n" + "="*60)
    print("WinTeach — Python Fundamentals Content Generator")
    print("="*60)
    print("\nSelect a CS50 Python week to generate artifacts for:")
    print()
    for i, name in enumerate(topic_names, 1):
        week_num = i - 1
        print(f"  {i}. Week {week_num} — {name}")
    print()

    while True:
        choice = input("\nEnter number (1-10): ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(topic_names):
            idx = int(choice) - 1
            name = topic_names[idx]
            url = CS50_NOTES_URLS[name]
            return url, name

        else:
            print(f"  Invalid choice. Enter a number between 1 and {len(topic_names)}.")


def main():
    while True:
        source, subtopic = pick_source()
        output = run_pipeline(source, subtopic)
        print_preview(output)
        save_output(output, subtopic)

        again = input("\nGenerate another topic? (y/n): ").strip().lower()
        if again != "y":
            print("\nDone. All outputs saved to outputs/ folder.")
            break


if __name__ == "__main__":
    main()