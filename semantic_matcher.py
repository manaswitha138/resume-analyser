from sentence_transformers import SentenceTransformer
import numpy as np


# ==========================================================
# LOAD SEMANTIC MODEL
# ==========================================================

_model = None


def load_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return _model


# ==========================================================
# TEXT SIMILARITY
# ==========================================================

def calculate_similarity(text1, text2):

    if not text1 or not text2:
        return 0.0

    model = load_model()

    embeddings = model.encode(
        [text1, text2],
        normalize_embeddings=True
    )

    similarity = float(
        np.dot(
            embeddings[0],
            embeddings[1]
        )
    )

    return round(
        max(0.0, min(1.0, similarity)),
        3
    )


# ==========================================================
# REQUIREMENT MATCHING
# ==========================================================

def match_requirements(
    resume_text,
    requirements,
    threshold=0.45
):

    if not resume_text or not requirements:
        return []

    model = load_model()

    resume_sentences = [
        sentence.strip()
        for sentence in resume_text.splitlines()
        if sentence.strip()
    ]

    if not resume_sentences:
        return []

    resume_embeddings = model.encode(
        resume_sentences,
        normalize_embeddings=True
    )

    requirement_embeddings = model.encode(
        requirements,
        normalize_embeddings=True
    )

    results = []

    for index, requirement in enumerate(requirements):

        similarities = np.dot(
            resume_embeddings,
            requirement_embeddings[index]
        )

        best_index = int(
            np.argmax(similarities)
        )

        best_score = float(
            similarities[best_index]
        )

        if best_score >= threshold:

            results.append({
                "requirement": requirement,
                "evidence": resume_sentences[best_index],
                "similarity": round(
                    best_score * 100,
                    1
                ),
                "status": "Strong Match"
            })

        else:

            results.append({
                "requirement": requirement,
                "evidence": None,
                "similarity": round(
                    best_score * 100,
                    1
                ),
                "status": "Weak / Missing"
            })

    return results


# ==========================================================
# OVERALL RESUME ↔ JOB MATCH
# ==========================================================

def calculate_semantic_job_match(
    resume_text,
    job_description
):

    similarity = calculate_similarity(
        resume_text,
        job_description
    )

    return {
        "score": round(similarity * 100),
        "level": get_match_level(
            similarity
        )
    }


# ==========================================================
# MATCH LEVEL
# ==========================================================

def get_match_level(similarity):

    if similarity >= 0.80:

        return "Excellent Match"

    elif similarity >= 0.65:

        return "Strong Match"

    elif similarity >= 0.50:

        return "Moderate Match"

    elif similarity >= 0.35:

        return "Weak Match"

    else:

        return "Low Match"