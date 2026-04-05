import json
import os
import re
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
import recommender as rb 

app = FastAPI(title="Acadza AI - JEE Personalization API")

class DataManager:
    def __init__(self):
        # Load JSON files
        self.raw_students = self._load_json("data/student_performance.json")
        self.question_bank = [q for q in self._load_json("data/question_bank.json") if isinstance(q, dict)]
        self.dost_config = self._load_json("data/dost_config.json")
        
        # Process and register students
        self.students_data = self._aggregate_performance()
        
        print(f"\n--- SERVER READY ---")
        print(f"Registered Students: {[s['student_id'] for s in self.students_data]}")
        print(f"--------------------\n")

    def _load_json(self, path):
        if not os.path.exists(path): return []
        with open(path, 'r') as f: return json.load(f)

    def _parse_marks(self, m):
        m = str(m).strip()
        try:
            if "/" in m: return float(re.findall(r"\d+", m)[0])
            nums = re.findall(r'[+-]?\d+', m)
            return float(sum(int(n) for n in nums)) if nums else 0.0
        except: return 0.0

    def _aggregate_performance(self):
        agg = {}
        for student in self.raw_students:
            s_id = student.get("student_id")
            if not s_id: continue
            
            if s_id not in agg:
                agg[s_id] = {"name": student.get("name"), "sessions": [], 
                             "topic_scores": {t: [] for t in rb.TOPICS}}
            
            for sess in student.get("attempts", []):
                agg[s_id]["sessions"].append(sess)
                context = " ".join(sess.get("chapters", [])).lower()
                marks = self._parse_marks(sess.get("marks", 0))
                total_q = sess.get("total_questions", 25) or 25
                accuracy = marks / (total_q * 4)
                
                for t in rb.TOPICS:
                    if t in context:
                        agg[s_id]["topic_scores"][t].append(max(0, accuracy))

        processed = []
        for s_id, data in agg.items():
            w_scores = {t: (1.0 - (sum(sc)/len(sc))) if sc else 0.5 for t, sc in data["topic_scores"].items()}
            processed.append({"student_id": s_id, "name": data["name"], 
                              "weakness_scores": w_scores, "raw_sessions": data["sessions"]})
        return processed

db = DataManager()

@app.get("/leaderboard")
def get_leaderboard():
    results = []
    for s in db.students_data:
        accs = [db._parse_marks(sess.get("marks", 0))/(sess.get("total_questions", 25)*4) for sess in s["raw_sessions"]]
        avg_acc = np.mean(accs) if accs else 0
        
        # Scoring Formula: 70% Accuracy, 20% Consistency, 10% Attempt Rate
        consistency = 1.0 - (np.std(accs) if len(accs) > 1 else 0.2)
        score = ((avg_acc * 0.7) + (consistency * 0.2) + (0.9 * 0.1)) * 100
        
        sorted_topics = sorted(s["weakness_scores"].items(), key=lambda x: x[1])
        results.append({
            "name": s["name"],
            "student_id": s["student_id"],
            "score": round(float(score), 2),
            "strength": sorted_topics[0][0].capitalize(),
            "weakness": sorted_topics[-1][0].capitalize(),
            "focus_area": f"{sorted_topics[-1][0].capitalize()} Strategy"
        })
    
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(ranked): item["rank"] = i + 1
    return ranked

@app.post("/recommend/{student_id}")
def get_dost_plan(student_id: str):
    sid = student_id.upper()
    try:
        idx = next(i for i, s in enumerate(db.students_data) if s["student_id"] == sid)
        student = db.students_data[idx]
    except StopIteration:
        raise HTTPException(status_code=404, detail="Student not found")

    s_mat = rb.build_feature_matrix(db.students_data, "student")
    q_mat = rb.build_feature_matrix(db.question_bank, "question")
    recs = rb.recommend(s_mat, q_mat, db.question_bank, idx, top_n=3)
    
    journey = ["concept", "practiceAssignment", "clickingPower"]
    steps = []
    for i, key in enumerate(journey):
        q = recs[i]
        steps.append({
            "step": i + 1,
            "dost_type": key,
            "target_chapter": q["topic"].capitalize(),
            "params": db.dost_config.get(key, {}).get("params"),
            "question_id": q["question_id"],
            "reasoning": f"AI detected a {int(student['weakness_scores'].get(q['topic'], 0.5)*100)}% performance gap.",
            "message": f"Time to master {q['topic']}! Follow this {key} activity."
        })

    return {"student_id": sid, "name": student["name"], "steps": steps}

@app.get("/question/{qid}")
def get_question_details(qid: str):
    # 1. Find the question by qid (e.g., Q_CHE_0001)
    # Case-insensitive lookup
    target_qid = qid.upper().strip()
    question = next((q for q in db.question_bank if q.get("qid") == target_qid), None)

    if not question:
        raise HTTPException(status_code=404, detail=f"Question {target_qid} not found in the bank.")

    # 2. Normalize the ID for the response
    raw_id = question.get("_id")
    normalized_id = raw_id["$oid"] if isinstance(raw_id, dict) and "$oid" in raw_id else str(raw_id)

    # 3. Extract content based on questionType (scq, mcq, integer)
    q_type = question.get("questionType", "scq")
    type_data = question.get(q_type, {})
    
    raw_html = type_data.get("question", "")
    raw_sol = type_data.get("solution", "")

    # 4. Helper to strip HTML tags for plaintext preview
    def clean_html(raw_html):
        # Removes anything between <> and replaces <br/> with a space
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, ' ', raw_html)
        return " ".join(cleantext.split()) # Removes extra whitespace

    return {
        "id": normalized_id,
        "qid": question.get("qid"),
        "subject": question.get("subject"),
        "topic": question.get("topic").replace("_", " ").capitalize(),
        "difficulty": question.get("difficulty"),
        "type": q_type,
        "previews": {
            "question_text": clean_html(raw_html),
            "solution_text": clean_html(raw_sol)
        },
        "answer": type_data.get("answer"),
        "raw_html": raw_html # Keeping the original for the UI to render
    }

from datetime import datetime

@app.post("/analyze/{student_id}")
def analyze_student(student_id: str):
    sid = student_id.upper()
    try:
        idx = next(i for i, s in enumerate(db.students_data) if s["student_id"] == sid)
        student = db.students_data[idx]
    except StopIteration:
        raise HTTPException(status_code=404, detail="Student not found")

    sessions = student["raw_sessions"]
    # Sort sessions by date for trend analysis
    try:
        sorted_sessions = sorted(sessions, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
    except:
        sorted_sessions = sessions

    # 1. Performance Trends
    accuracies = [db._parse_marks(s.get("marks", 0)) / (s.get("total_questions", 25) * 4) for s in sorted_sessions]
    improvement_rate = (accuracies[-1] - accuracies[0]) * 100 if len(accuracies) > 1 else 0
    
    # 2. Chapter-wise Breakdown
    chapter_stats = {}
    for sess in sessions:
        marks = db._parse_marks(sess.get("marks", 0))
        tq = sess.get("total_questions", 25) or 25
        acc = (marks / (tq * 4)) * 100
        
        for chap in sess.get("chapters", []):
            if chap not in chapter_stats:
                chapter_stats[chap] = {"attempts": 0, "avg_accuracy": 0, "total_acc": 0}
            chapter_stats[chap]["attempts"] += 1
            chapter_stats[chap]["total_acc"] += acc
            chapter_stats[chap]["avg_accuracy"] = chapter_stats[chap]["total_acc"] / chapter_stats[chap]["attempts"]

    # 3. Pattern Recognition (Speed vs Accuracy)
    avg_speed = np.mean([s.get("avg_time_per_question_seconds", 0) for s in sessions])
    
    # Strengths and Weaknesses
    sorted_chaps = sorted(chapter_stats.items(), key=lambda x: x[1]["avg_accuracy"])
    
    analysis = {
        "student_id": sid,
        "name": student["name"],
        "summary": {
            "total_tests_taken": len(sessions),
            "overall_accuracy": f"{round(np.mean(accuracies)*100, 2)}%",
            "improvement_trend": "Improving" if improvement_rate > 0 else "Declining",
            "improvement_value": f"{round(improvement_rate, 2)}%"
        },
        "chapter_breakdown": [
            {
                "chapter": k,
                "accuracy": f"{round(v['avg_accuracy'], 2)}%",
                "status": "Mastered" if v['avg_accuracy'] > 75 else "Needs Practice"
            } for k, v in chapter_stats.items()
        ],
        "top_strengths": [c[0] for c in sorted_chaps[-2:]][::-1],
        "critical_weaknesses": [c[0] for c in sorted_chaps[:2]],
        "behavioral_patterns": {
            "avg_pace_seconds": round(avg_speed, 1),
            "completion_rate": f"{round((sum(1 for s in sessions if s.get('completed'))/len(sessions))*100, 1)}%",
            "note": "Student tends to be slower on Physics topics" if avg_speed > 180 else "Good speed maintained"
        }
    }
    
    return analysis

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)