"""Build an ISOLATED evaluation set. NEVER fed to teachers as training seeds.

Held-out questions across categories. Some are original; the file marks them so the
training generator never reads them. Public-benchmark-style items are original phrasings
(not copied benchmarks) to avoid contamination issues.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "distilled_corpus" / "evaluation" / "evaluation_set.jsonl"

EVAL = [
    # general knowledge
    {"id":"eval-gk-01","category":"general_knowledge","prompt":"What is the capital of Canada?","answer":"Ottawa","type":"short_answer"},
    {"id":"eval-gk-02","category":"general_knowledge","prompt":"In which decade did the Berlin Wall fall?","answer":"1989 (late 1980s)","type":"short_answer"},
    {"id":"eval-gk-03","category":"general_knowledge","prompt":"Name the author of 'War and Peace'.","answer":"Leo Tolstoy","type":"short_answer"},
    # science
    {"id":"eval-sci-01","category":"science","prompt":"What is the chemical symbol for gold?","answer":"Au","type":"short_answer"},
    {"id":"eval-sci-02","category":"science","prompt":"State the second law of thermodynamics in one sentence.","answer":"entropy of an isolated system never decreases","type":"graded"},
    {"id":"eval-sci-03","category":"science","prompt":"What organelle is responsible for protein synthesis in a cell?","answer":"ribosome","type":"short_answer"},
    # math
    {"id":"eval-math-01","category":"math","prompt":"What is the derivative of sin(x)?","answer":"cos(x)","type":"short_answer"},
    {"id":"eval-math-02","category":"math","prompt":"Compute the determinant of [[2,1],[3,4]].","answer":"5","type":"numeric"},
    {"id":"eval-math-03","category":"math","prompt":"Solve x^2 - 5x + 6 = 0.","answer":"x=2 and x=3","type":"numeric"},
    # programming
    {"id":"eval-code-01","category":"programming","prompt":"Write a Python one-liner to reverse the string 'hello'.","answer":"'hello'[::-1] == 'olleh'","type":"code"},
    {"id":"eval-code-02","category":"programming","prompt":"What is the time complexity of binary search?","answer":"O(log n)","type":"short_answer"},
    {"id":"eval-code-03","category":"programming","prompt":"Write a Python function to compute the factorial of n iteratively.","answer":"def fact(n):\n r=1\n for i in range(2,n+1): r*=i\n return r","type":"code"},
    # reasoning
    {"id":"eval-rea-01","category":"reasoning","prompt":"If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?","answer":"5 minutes","type":"numeric"},
    {"id":"eval-rea-02","category":"reasoning","prompt":"A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball?","answer":"$0.05","type":"numeric"},
    # technology / long context
    {"id":"eval-tech-01","category":"technology","prompt":"Explain the difference between TCP and UDP in 3 sentences.","answer":"graded: connection-oriented vs connectionless, reliability, ordering","type":"graded"},
    {"id":"eval-lc-01","category":"long_context","prompt":"Read a long document and answer: (placeholder for long-context retrieval test).","answer":"graded","type":"long_context"},
]

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for e in EVAL:
            e["isolated"] = True
            e["never_in_training"] = True
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(EVAL)} eval items -> {OUT}")

if __name__ == "__main__":
    main()
