import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Standardized JEE topics
TOPICS = [
    "mechanics", "thermodynamics", "electrostatics", "optics",
    "kinematics", "laws of motion", "modern_physics", 
    "organic chemistry", "inorganic chemistry", "physical_chemistry",
    "chemical bonding", "kinetics", "algebra", "calculus"
]
TOPIC_TO_IDX = {t: i for i, t in enumerate(TOPICS)}

def build_feature_matrix(records: list[dict], record_type: str = "student") -> np.ndarray:
    """Build a normalized feature matrix with safety guards for missing data."""
    n_records = len(records)
    matrix = np.zeros((n_records, len(TOPICS)))

    if record_type == "student":
        for i, rec in enumerate(records):
            for topic, score in rec.get("weakness_scores", {}).items():
                t_lower = topic.lower()
                if t_lower in TOPIC_TO_IDX:
                    matrix[i, TOPIC_TO_IDX[t_lower]] = score
    else:
        for i, rec in enumerate(records):
            topic = str(rec.get("topic", "")).lower()
            
            # SAFETY GUARD: Handle NoneType or missing difficulty
            raw_diff = rec.get("difficulty")
            diff_val = raw_diff if raw_diff is not None else 3
            
            try:
                weight = float(diff_val) / 5.0
            except (ValueError, TypeError):
                weight = 0.6  # Default to 3/5 weight
                
            if topic in TOPIC_TO_IDX:
                matrix[i, TOPIC_TO_IDX[topic]] = weight

    return normalize(matrix, axis=1, norm="l2")

def recommend(student_matrix: np.ndarray, question_matrix: np.ndarray,
              questions: list[dict], student_idx: int, top_n: int = 10) -> list[dict]:
    """Finds questions matching the student's unique weakness profile."""
    # Calculate cohort baseline to find unique gaps
    cohort_baseline = student_matrix.mean(axis=0)
    student_profile = student_matrix[student_idx] - cohort_baseline

    profile_norm = np.linalg.norm(student_profile)
    if profile_norm > 1e-10:
        student_profile = student_profile / profile_norm
    else:
        student_profile = student_matrix[student_idx]

    similarities = cosine_similarity(student_profile.reshape(1, -1), question_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    return [{
        "question_id": questions[idx].get("qid"),
        "topic": questions[idx].get("topic"),
        "difficulty": questions[idx].get("difficulty"),
        "score": round(float(similarities[idx]), 4)
    } for idx in top_indices]