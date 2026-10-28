"""
tests/eval.py  –  LexPal Evaluation Script
===========================================
Runs all 30 test questions against the /ask endpoint and measures:
  - Answer correctness (keyword match as proxy)
  - Citation accuracy (expected section present in sources)
  - Hallucination rate (answer contains "I don't know" or invented sections)

Usage:
    python tests/eval.py
    python tests/eval.py --api http://localhost:8000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def load_questions(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def ask(api_base: str, question: str, language: str = "English") -> dict | None:
    try:
        resp = requests.post(
            f"{api_base}/ask",
            json={"question": question, "language": language},
            timeout=120,
        )
        if resp.ok:
            return resp.json()
        print(f"  ⚠️  API error {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
        return None


def check_keywords(answer: str, keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def check_citation(sources: list[dict], expected_section: str, expected_act: str) -> bool:
    for s in sources:
        if expected_act.lower() in s.get("act", "").lower():
            section = s.get("section", "").lower()
            # Match section number loosely
            sec_num = expected_section.lower().replace("section ", "").strip()
            if sec_num in section:
                return True
    return False


def is_refusing(answer: str) -> bool:
    refusal_phrases = [
        "don't have enough information",
        "not in my database",
        "consult a licensed lawyer",
        "i don't know",
        "cannot find",
        "not available in my database",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in refusal_phrases)


def main():
    parser = argparse.ArgumentParser(description="LexPal evaluation")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--language", default="English", help="Reply language")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between requests")
    args = parser.parse_args()

    questions_file = Path(__file__).parent / "test_questions.json"
    questions = load_questions(questions_file)
    total = len(questions)

    print(f"\n{'='*60}")
    print(f"  LexPal Evaluation Suite  –  {total} questions")
    print(f"  API: {args.api}  |  Language: {args.language}")
    print(f"{'='*60}\n")

    # ── Check API health ────────────────────────────────────────────────────
    try:
        health = requests.get(f"{args.api}/health", timeout=5).json()
        db_chunks = health.get("db_chunks", 0)
        print(f"  ✅ Backend online  |  DB chunks: {db_chunks}\n")
        if db_chunks == 0:
            print("  ⚠️  WARNING: DB is empty. Run ingestion first. Results will be poor.\n")
    except Exception as e:
        print(f"  ❌ Backend unreachable: {e}\n")
        sys.exit(1)

    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i:02d}/{total}] {q['id']} — {q['question'][:70]}…")
        data = ask(args.api, q["question"], args.language)
        time.sleep(args.delay)

        if data is None:
            results.append({**q, "answered": False, "keyword_match": False,
                            "citation_match": False, "hallucinated": False})
            print("       → FAILED (no response)\n")
            continue

        answer  = data.get("answer", "")
        sources = data.get("sources", [])

        answered       = bool(answer.strip()) and not is_refusing(answer)
        keyword_match  = check_keywords(answer, q["expected_keywords"])
        citation_match = check_citation(sources, q["expected_section"], q["expected_act"])

        # Hallucination: mentioned a section not in sources (very rough heuristic)
        import re
        cited_sections = {s.get("section", "").lower() for s in sources}
        answer_sections = set(re.findall(r"section\s+\d+[a-z]?", answer.lower()))
        hallucinated = bool(answer_sections - cited_sections) and not is_refusing(answer)

        status = "✅" if keyword_match and citation_match else ("⚠️" if keyword_match else "❌")
        print(f"       → {status}  keyword={keyword_match}  citation={citation_match}  hallucinated={hallucinated}")
        print(f"          Answer preview: {answer[:120].strip()}\n")

        results.append({
            **q,
            "answered":       answered,
            "keyword_match":  keyword_match,
            "citation_match": citation_match,
            "hallucinated":   hallucinated,
            "answer_preview": answer[:200],
        })

    # ── Summary ─────────────────────────────────────────────────────────────
    answered_count  = sum(1 for r in results if r["answered"])
    keyword_count   = sum(1 for r in results if r["keyword_match"])
    citation_count  = sum(1 for r in results if r["citation_match"])
    hallucin_count  = sum(1 for r in results if r["hallucinated"])

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS  ({total} questions)")
    print(f"{'='*60}")
    print(f"  Answer rate        : {answered_count}/{total} = {answered_count/total*100:.1f}%")
    print(f"  Keyword accuracy   : {keyword_count}/{total} = {keyword_count/total*100:.1f}%")
    print(f"  Citation accuracy  : {citation_count}/{total} = {citation_count/total*100:.1f}%")
    print(f"  Hallucination rate : {hallucin_count}/{total} = {hallucin_count/total*100:.1f}%")
    print(f"{'='*60}")

    target_met = keyword_count / total >= 0.80
    print(f"\n  Target (≥80% keyword accuracy): {'✅ MET' if target_met else '❌ NOT MET'}")
    print(f"  Zero hallucinations target:     {'✅ MET' if hallucin_count == 0 else f'❌ {hallucin_count} hallucinations found'}")

    # ── Save report ──────────────────────────────────────────────────────────
    report_path = Path(__file__).parent / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "answered": answered_count,
                "keyword_accuracy_pct": round(keyword_count / total * 100, 1),
                "citation_accuracy_pct": round(citation_count / total * 100, 1),
                "hallucination_rate_pct": round(hallucin_count / total * 100, 1),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Full report saved to: {report_path}\n")


if __name__ == "__main__":
    main()
