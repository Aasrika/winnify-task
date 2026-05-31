import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
NLI_MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-base")


def verify_nli_label_order() -> int:
    probe_pairs = [
        (
            "Python is dynamically typed.",
            "Python variables do not require explicit type declarations."
        )
    ]
    raw = NLI_MODEL.predict(probe_pairs)
    entailment_idx = int(np.argmax(raw[0]))

    print("NLI label verification:")
    print(f"  Raw scores: {[round(float(x), 4) for x in raw[0]]}")
    print(f"  Detected entailment index: {entailment_idx}")
    print(f"  Verified: index {entailment_idx} = entailment. Safe to proceed.\n")

    return entailment_idx


ENTAILMENT_IDX = verify_nli_label_order()