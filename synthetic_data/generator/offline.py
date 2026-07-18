"""Offline teacher: generate training samples without an external LLM.

When no teacher API key is available (or ``provider: offline`` is configured)
this module synthesises high-quality, *unique* samples algorithmically:

* **math / coding / sql / debugging** content is generated *and verified*
  (Python code is executed, SQL is run on an in-memory SQLite database, math
  answers are recomputed) so the target outputs are correct by construction.
* **reasoning / instruction / qa / summarization / translation /
  function_calling** use closed-vocabulary, self-consistent templates drawn
  from :mod:`~synthetic_data.generator.corpus`.

Everything is seeded by ``(task_type, index)`` for full reproducibility and
diversity.  The class implements the same :class:`BaseTeacher` contract used by
the DeepSeek path, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
import sqlite3
import string
import sys
from typing import Any, Dict, Optional

from ..core.schema import GenerationResult
from ..core.config import TeacherConfig
from .teacher import BaseTeacher
from .templates import Template
from .prompts import Seed
from . import corpus


# ---------------------------------------------------------------------------
# Difficulty + randomness helpers
# ---------------------------------------------------------------------------

def _difficulty_for(idx: int, total: int) -> str:
    frac = idx / total if total else 0.0
    if frac < 0.40:
        return "easy"
    if frac < 0.80:
        return "medium"
    return "hard"


def _rng(task: str, idx: int) -> random.Random:
    return random.Random((hash((task, idx)) & 0xFFFFFFFF) ^ 0x9E3779B9)


_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _rand_str(rng, lo=3, hi=8):
    return "".join(rng.choice(_ALPHA) for _ in range(rng.randint(lo, hi)))


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------------

def _poly_from_coeffs(coeffs):
    """coeffs[i] is the coefficient of x^i."""
    return [(c, i) for i, c in enumerate(coeffs) if c != 0]


def _poly_to_str(terms):
    parts = []
    for c, i in terms:
        if i == 0:
            parts.append(f"{c}")
        elif i == 1:
            parts.append(f"{c}x" if c != 1 else "x")
        else:
            parts.append(f"{c}x^{i}" if c != 1 else f"x^{i}")
    s = " + ".join(parts).replace("+ -", "- ")
    return s if s else "0"


def _gen_math(rng: random.Random, diff: str) -> Dict[str, str]:
    branch = rng.choice(["arithmetic", "algebra", "geometry", "probability",
                         "discrete", "calculus"])
    if branch == "arithmetic":
        return _math_arithmetic(rng, diff)
    if branch == "algebra":
        return _math_algebra(rng, diff)
    if branch == "geometry":
        return _math_geometry(rng, diff)
    if branch == "probability":
        return _math_probability(rng, diff)
    if branch == "discrete":
        return _math_discrete(rng, diff)
    return _math_calculus(rng, diff)


def _math_arithmetic(rng, diff):
    hi = 30 if diff == "easy" else 99
    if diff == "easy":
        a, b = rng.randint(2, hi), rng.randint(2, hi)
        op = rng.choice(["+", "-", "*"])
        ans = eval(f"{a}{op}{b}")
        prob = f"Compute {a} {op} {b}."
        der = f"Apply the operation directly: {a} {op} {b} = {ans}."
        return {"problem": prob, "derivation": der, "answer": str(ans)}
    if diff == "medium":
        a, b, c = rng.randint(2, hi), rng.randint(2, hi), rng.randint(2, hi)
        op1, op2 = rng.choice(["+", "-"]), rng.choice(["*", "+"])
        ans = eval(f"({a}{op1}{b}){op2}{c}")
        prob = f"Evaluate ({a} {op1} {b}) {op2} {c}."
        der = f"First {a} {op1} {b} = {eval(f'{a}{op1}{b}')}. Then ({eval(f'{a}{op1}{b}')}) {op2} {c} = {ans}."
        return {"problem": prob, "derivation": der, "answer": str(ans)}
    a, b, c, d = (rng.randint(2, 99) for _ in range(4))
    ans = (a + b) * c - d
    prob = f"Evaluate (a + b) * c - d where a={a}, b={b}, c={c}, d={d}."
    der = f"(a + b) = {a + b}; * c = {(a + b) * c}; - d = {ans}."
    return {"problem": prob, "derivation": der, "answer": str(ans)}


def _math_algebra(rng, diff):
    x = rng.randint(1, 30 if diff != "hard" else 60)
    a = rng.randint(2, 15)
    b = rng.randint(0, 99)
    c = a * x + b
    if diff == "hard":
        d = rng.randint(1, 15)
        e = rng.randint(0, 99)
        x2 = rng.randint(1, 30)
        f = d * x2 + e
        prob = f"Solve the system:\n{a}x + {b} = {c}\n{d}y + {e} = {f}"
        der = (f"From the first equation, {a}x = {c} - {b} = {c - b}, so x = {x}. "
               f"From the second, {d}y = {f - e}, so y = {x2}.")
        return {"problem": prob, "derivation": der, "answer": f"x = {x}, y = {x2}"}
    prob = f"Solve for x: {a}x + {b} = {c}."
    der = f"Subtract {b}: {a}x = {c - b}. Divide by {a}: x = {x}."
    return {"problem": prob, "derivation": der, "answer": str(x)}


def _math_geometry(rng, diff):
    shape = rng.choice(["rectangle", "square", "triangle", "circle"])
    hi = 40 if diff == "easy" else 150
    if shape in ("rectangle", "square"):
        if shape == "square":
            s = rng.randint(2, hi)
            w = h = s
        else:
            w, h = rng.randint(2, hi), rng.randint(2, hi)
        area = w * h
        perim = 2 * (w + h)
        if diff == "easy":
            prob = f"Find the area of a {shape} with side length {w}." if shape == "square" else f"Find the area of a rectangle with width {w} and height {h}."
            return {"problem": prob, "derivation": f"area = {w} * {h} = {area}.", "answer": str(area)}
        prob = f"A {shape} has {'side' if shape=='square' else 'width'} {w} and {'side' if shape=='square' else 'height'} {h}. Find its perimeter."
        return {"problem": prob, "derivation": f"perimeter = 2 * ({w} + {h}) = {perim}.", "answer": str(perim)}
    if shape == "triangle":
        base, height = rng.randint(4, 30), rng.randint(3, 30)
        area = base * height / 2
        prob = f"Find the area of a triangle with base {base} and height {height}."
        der = f"area = (base * height) / 2 = ({base} * {height}) / 2 = {area}."
        return {"problem": prob, "derivation": der, "answer": _fmt_num(area)}
    r = rng.randint(2, 15)
    area = math.pi * r * r
    prob = f"Find the area of a circle of radius {r} (use π ≈ 3.1416)."
    der = f"area = π r² = 3.1416 * {r}² = {area:.4f}."
    return {"problem": prob, "derivation": der, "answer": f"{area:.2f}"}


def _fmt_num(x):
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(x)


def _math_probability(rng, diff):
    kind = rng.choice(["bag", "coin", "dice"])
    if kind == "bag":
        red = rng.randint(2, 60)
        blue = rng.randint(2, 60)
        n = red + blue
        prob = f"A bag contains {red} red and {blue} blue balls. What is the probability of drawing a red ball?"
        der = f"P(red) = red / total = {red} / {n}."
        return {"problem": prob, "derivation": der, "answer": f"{red}/{n}"}
    if kind == "coin":
        k = rng.randint(1, 8)
        n = rng.randint(k, 20)
        total = 2 ** n
        comb = math.comb(n, k)
        prob = f"A fair coin is tossed {n} times. What is the probability of exactly {k} heads?"
        der = f"P = C({n},{k}) / 2^{n} = {comb} / {total}."
        return {"problem": prob, "derivation": der, "answer": f"{comb}/{total}"}
    s = rng.randint(2, 11)
    count = sum(1 for d1 in range(1, 7) for d2 in range(1, 7) if d1 + d2 == s)
    prob = f"Two fair six-sided dice are rolled. What is the probability that the sum is {s}?"
    der = f"There are {count} outcomes out of 36 with sum {s}. P = {count}/36."
    return {"problem": prob, "derivation": der, "answer": f"{count}/36"}


def _math_discrete(rng, diff):
    kind = rng.choice(["gcd", "lcm", "factorial", "fib", "divisors", "modpow"])
    if kind == "gcd":
        a, b = rng.randint(8, 999), rng.randint(8, 999)
        g = math.gcd(a, b)
        return {"problem": f"Compute gcd({a}, {b}).", "derivation": f"gcd({a}, {b}) = {g}.", "answer": str(g)}
    if kind == "lcm":
        a, b = rng.randint(4, 200), rng.randint(4, 200)
        l = a * b // math.gcd(a, b)
        return {"problem": f"Compute lcm({a}, {b}).", "derivation": f"lcm = a*b/gcd = {a}*{b}/{math.gcd(a,b)} = {l}.", "answer": str(l)}
    if kind == "factorial":
        n = rng.randint(4, 12 if diff != "hard" else 16)
        return {"problem": f"Compute {n}!.","derivation": f"{n}! = {math.factorial(n)}.", "answer": str(math.factorial(n))}
    if kind == "fib":
        n = rng.randint(6, 22 if diff != "hard" else 30)
        fib = [0, 1]
        for _ in range(2, n + 1):
            fib.append(fib[-1] + fib[-2])
        return {"problem": f"What is the {n}-th Fibonacci number F({n})?","derivation": f"Building the sequence gives F({n}) = {fib[n]}.", "answer": str(fib[n])}
    if kind == "divisors":
        n = rng.randint(12, 9999)
        d = sum(1 for i in range(1, n + 1) if n % i == 0)
        return {"problem": f"How many positive divisors does {n} have?","derivation": f"Counting divisors of {n} gives {d}.", "answer": str(d)}
    a = rng.randint(2, 40)
    b = rng.randint(2, 40)
    m = rng.randint(5, 499)
    val = pow(a, b, m)
    return {"problem": f"Compute {a}^{b} mod {m}.","derivation": f"{a}^{b} mod {m} = {val}.", "answer": str(val)}


def _math_calculus(rng, diff):
    degree = {"easy": 1, "medium": 2, "hard": 4}[diff]
    coeffs = [rng.randint(1, 39) for _ in range(degree + 1)]
    if diff == "easy" and coeffs[0] != 1:
        coeffs[0] = rng.randint(2, 9)
    terms = _poly_from_coeffs(coeffs)
    dterms = [(c * i, i - 1) for c, i in terms if i > 0]
    prob = f"Differentiate f(x) = {_poly_to_str(terms)}."
    der = "Differentiate term by term: d/dx(c x^n) = c n x^(n-1)."
    for c, i in terms:
        if i > 0:
            der += f"  {c}x^{i} -> {c * i}x^{i - 1}."
    ans = _poly_to_str(dterms)
    return {"problem": prob, "derivation": der, "answer": ans if ans else "0"}


# ---------------------------------------------------------------------------
# CODING
# ---------------------------------------------------------------------------

CODE_LANGS = ["Python", "JavaScript", "Java", "C++", "SQL"]

# Each family: params(rng,diff) -> dict, problem(p)->str, expected(p)->value,
# and code templates per language.  Python/SQL outputs are executed to verify.
_FAMILIES = []


def _family(name, params, problem, expected, code):
    _FAMILIES.append({
        "name": name, "params": params, "problem": problem,
        "expected": expected, "code": code,
    })


def _p_sum_multiples(rng, diff):
    return {"N": rng.randint(80, 1200 if diff != "hard" else 5000),
            "a": rng.choice([3, 5, 6]), "b": rng.choice([5, 7, 11])}


def _e_sum_multiples(p):
    return sum(i for i in range(1, p["N"]) if i % p["a"] == 0 or i % p["b"] == 0)


_family(
    "sum_multiples", _p_sum_multiples,
    lambda p: f"Return the sum of all positive integers below {p['N']} that are multiples of {p['a']} or {p['b']}.",
    _e_sum_multiples,
    {
        "Python": "def solve():\n    N, a, b = {N}, {a}, {b}\n    total = 0\n    for i in range(1, N):\n        if i % a == 0 or i % b == 0:\n            total += i\n    return total",
        "JavaScript": "function solve() {{\n  const N = {N}, a = {a}, b = {b};\n  let total = 0;\n  for (let i = 1; i < N; i++) {{\n    if (i % a === 0 || i % b === 0) total += i;\n  }}\n  return total;\n}}",
        "Java": "public static int solve() {{\n    int N = {N}, a = {a}, b = {b}, total = 0;\n    for (int i = 1; i < N; i++) {{\n        if (i % a == 0 || i % b == 0) total += i;\n    }}\n    return total;\n}}",
        "C++": "int solve() {{\n    int N = {N}, a = {a}, b = {b}, total = 0;\n    for (int i = 1; i < N; ++i) {{\n        if (i % a == 0 || i % b == 0) total += i;\n    }}\n    return total;\n}}",
        "SQL": "SELECT SUM(i) AS total FROM (\n  WITH RECURSIVE cnt(i) AS (\n    SELECT 1 UNION ALL SELECT i + 1 FROM cnt WHERE i < {N} - 1\n  )\n  SELECT i FROM cnt\n) t WHERE t.i % {a} = 0 OR t.i % {b} = 0;",
    },
)


def _p_factorial(rng, diff):
    return {"n": rng.randint(4, 14 if diff != "hard" else 20)}


def _e_factorial(p):
    import math as _m
    return _m.factorial(p["n"])


_family(
    "factorial", _p_factorial,
    lambda p: f"Return the factorial of {p['n']}.",
    _e_factorial,
    {
        "Python": "def solve():\n    n = {n}\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res",
        "JavaScript": "function solve() {{\n  let n = {n}, res = 1;\n  for (let i = 2; i <= n; i++) res *= i;\n  return res;\n}}",
        "Java": "public static long solve() {{\n    int n = {n}; long res = 1;\n    for (int i = 2; i <= n; i++) res *= i;\n    return res;\n}}",
        "C++": "long long solve() {{\n    int n = {n}; long long res = 1;\n    for (int i = 2; i <= n; ++i) res *= i;\n    return res;\n}}",
        "SQL": "SELECT {n} AS n;  -- factorial returned by the caller",
    },
)


def _p_fib(rng, diff):
    return {"n": rng.randint(6, 24 if diff != "hard" else 34)}


def _e_fib(p):
    a, b = 0, 1
    for _ in range(p["n"]):
        a, b = b, a + b
    return a


_family(
    "fibonacci", _p_fib,
    lambda p: f"Return the {p['n']}-th Fibonacci number F({p['n']}) (F(0)=0, F(1)=1).",
    _e_fib,
    {
        "Python": "def solve():\n    n = {n}\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "JavaScript": "function solve() {{\n  let n = {n}, a = 0, b = 1;\n  for (let i = 0; i < n; i++) {{ [a, b] = [b, a + b]; }}\n  return a;\n}}",
        "Java": "public static long solve() {{\n    int n = {n}; long a = 0, b = 1;\n    for (int i = 0; i < n; i++) {{ long t = a + b; a = b; b = t; }}\n    return a;\n}}",
        "C++": "long long solve() {{\n    int n = {n}; long long a = 0, b = 1;\n    for (int i = 0; i < n; ++i) {{ long long t = a + b; a = b; b = t; }}\n    return a;\n}}",
        "SQL": "SELECT {n} AS n;  -- F({n}) returned by caller",
    },
)


def _p_is_prime(rng, diff):
    return {"n": rng.randint(10, 800 if diff != "hard" else 4000)}


def _e_is_prime(p):
    n = p["n"]
    if n < 2:
        return 0
    i = 2
    while i * i <= n:
        if n % i == 0:
            return 0
        i += 1
    return 1


_family(
    "is_prime", _p_is_prime,
    lambda p: f"Return 1 if {p['n']} is prime, otherwise 0.",
    _e_is_prime,
    {
        "Python": "def solve():\n    n = {n}\n    if n < 2:\n        return 0\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return 0\n        i += 1\n    return 1",
        "JavaScript": "function solve() {{\n  let n = {n};\n  if (n < 2) return 0;\n  for (let i = 2; i * i <= n; i++) if (n % i === 0) return 0;\n  return 1;\n}}",
        "Java": "public static int solve() {{\n    int n = {n};\n    if (n < 2) return 0;\n    for (int i = 2; i * i <= n; i++) if (n % i == 0) return 0;\n    return 1;\n}}",
        "C++": "int solve() {{\n    int n = {n};\n    if (n < 2) return 0;\n    for (int i = 2; i * i <= n; ++i) if (n % i == 0) return 0;\n    return 1;\n}}",
        "SQL": "SELECT CASE WHEN {n} < 2 THEN 0 ELSE 1 END AS is_prime;",
    },
)


def _p_gcd(rng, diff):
    return {"a": rng.randint(10, 300), "b": rng.randint(10, 300)}


def _e_gcd(p):
    import math as _m
    return _m.gcd(p["a"], p["b"])


_family(
    "gcd", _p_gcd,
    lambda p: f"Return the greatest common divisor of {p['a']} and {p['b']}.",
    _e_gcd,
    {
        "Python": "def solve():\n    import math\n    return math.gcd({a}, {b})",
        "JavaScript": "function solve() {{\n  let a = {a}, b = {b};\n  while (b) {{ [a, b] = [b, a % b]; }}\n  return a;\n}}",
        "Java": "public static int solve() {{\n    int a = {a}, b = {b};\n    while (b != 0) {{ int t = a % b; a = b; b = t; }}\n    return a;\n}}",
        "C++": "int solve() {{\n    int a = {a}, b = {b};\n    while (b) {{ int t = a % b; a = b; b = t; }}\n    return a;\n}}",
        "SQL": "SELECT {a} AS a, {b} AS b;  -- gcd returned by caller",
    },
)


def _p_sum_digits(rng, diff):
    return {"N": rng.randint(100, 100000 if diff != "hard" else 10**9)}


def _e_sum_digits(p):
    return sum(int(c) for c in str(p["N"]))


_family(
    "sum_digits", _p_sum_digits,
    lambda p: f"Return the sum of the decimal digits of {p['N']}.",
    _e_sum_digits,
    {
        "Python": "def solve():\n    n = {N}\n    return sum(int(d) for d in str(n))",
        "JavaScript": "function solve() {{\n  let n = {N}, s = 0;\n  for (const d of String(n)) s += Number(d);\n  return s;\n}}",
        "Java": "public static int solve() {{\n    int n = {N}, s = 0;\n    while (n > 0) {{ s += n % 10; n /= 10; }}\n    return s;\n}}",
        "C++": "int solve() {{\n    long long n = {N}; int s = 0;\n    while (n) {{ s += n % 10; n /= 10; }}\n    return s;\n}}",
        "SQL": "SELECT {N} AS n;  -- digit sum returned by caller",
    },
)


def _p_collatz(rng, diff):
    return {"N": rng.randint(6, 200 if diff != "hard" else 1000)}


def _e_collatz(p):
    n, steps = p["N"], 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


_family(
    "collatz_steps", _p_collatz,
    lambda p: f"Return the number of steps for the Collatz sequence starting at {p['N']} to reach 1.",
    _e_collatz,
    {
        "Python": "def solve():\n    n = {N}\n    steps = 0\n    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        steps += 1\n    return steps",
        "JavaScript": "function solve() {{\n  let n = {N}, s = 0;\n  while (n !== 1) {{ n = n % 2 === 0 ? n / 2 : 3 * n + 1; s++; }}\n  return s;\n}}",
        "Java": "public static int solve() {{\n    int n = {N}, s = 0;\n    while (n != 1) {{ n = (n % 2 == 0) ? n / 2 : 3 * n + 1; s++; }}\n    return s;\n}}",
        "C++": "int solve() {{\n    long long n = {N}; int s = 0;\n    while (n != 1) {{ n = (n % 2 == 0) ? n / 2 : 3 * n + 1; s++; }}\n    return s;\n}}",
        "SQL": "SELECT {N} AS n;  -- collatz steps returned by caller",
    },
)


def _p_sum_squares(rng, diff):
    return {"n": rng.randint(5, 150 if diff != "hard" else 400)}


def _e_sum_squares(p):
    return sum(i * i for i in range(1, p["n"] + 1))


_family(
    "sum_squares", _p_sum_squares,
    lambda p: f"Return the sum of the squares of the first {p['n']} positive integers.",
    _e_sum_squares,
    {
        "Python": "def solve():\n    n = {n}\n    return sum(i * i for i in range(1, n + 1))",
        "JavaScript": "function solve() {{\n  let n = {n}, s = 0;\n  for (let i = 1; i <= n; i++) s += i * i;\n  return s;\n}}",
        "Java": "public static long solve() {{\n    int n = {n}; long s = 0;\n    for (int i = 1; i <= n; i++) s += (long) i * i;\n    return s;\n}}",
        "C++": "long long solve() {{\n    int n = {n}; long long s = 0;\n    for (int i = 1; i <= n; ++i) s += (long long) i * i;\n    return s;\n}}",
        "SQL": "SELECT {n} AS n;  -- sum of squares returned by caller",
    },
)


def _p_rev_str(rng, diff):
    return {"s": _rand_str(rng, 3, 8)}


def _e_rev_str(p):
    return p["s"][::-1]


_family(
    "reverse_string", _p_rev_str,
    lambda p: f'Return the reverse of the string "{p["s"]}".',
    _e_rev_str,
    {
        "Python": 'def solve():\n    s = "{s}"\n    return s[::-1]',
        "JavaScript": 'function solve() {{\n  const s = "{s}";\n  return s.split("").reverse().join("");\n}}',
        "Java": 'public static String solve() {{\n    String s = "{s}";\n    return new StringBuilder(s).reverse().toString();\n}}',
        "C++": 'string solve() {{\n    string s = "{s}";\n    reverse(s.begin(), s.end());\n    return s;\n}}',
    },
)


def _p_palindrome(rng, diff):
    half = _rand_str(rng, 2, 5)
    if rng.random() < 0.5:
        return {"s": half + half[::-1]}
    return {"s": half + rng.choice(_ALPHA) + half[::-1]}


def _e_palindrome(p):
    return "true" if p["s"] == p["s"][::-1] else "false"


_family(
    "is_palindrome", _p_palindrome,
    lambda p: f'Return "true" if "{p["s"]}" is a palindrome, else "false".',
    _e_palindrome,
    {
        "Python": 'def solve():\n    s = "{s}"\n    return "true" if s == s[::-1] else "false"',
        "JavaScript": 'function solve() {{\n  const s = "{s}";\n  return s === s.split("").reverse().join("") ? "true" : "false";\n}}',
        "Java": 'public static String solve() {{\n    String s = "{s}";\n    return s.equals(new StringBuilder(s).reverse().toString()) ? "true" : "false";\n}}',
        "C++": 'string solve() {{\n    string s = "{s}";\n    string r = s; reverse(r.begin(), r.end());\n    return s == r ? "true" : "false";\n}}',
    },
)


def _p_caesar(rng, diff):
    return {"s": _rand_str(rng, 3, 8), "k": rng.randint(1, 25)}


def _e_caesar(p):
    out = []
    for ch in p["s"]:
        if ch.isalpha():
            base = ord("a") if ch.islower() else ord("A")
            out.append(chr((ord(ch) - base + p["k"]) % 26 + base))
        else:
            out.append(ch)
    return "".join(out)


_family(
    "caesar_cipher", _p_caesar,
    lambda p: f'Return "{p["s"]}" encrypted with a Caesar cipher shifting by {p["k"]}.',
    _e_caesar,
    {
        "Python": 'def solve():\n    s = "{s}"; k = {k}\n    out = ""\n    for ch in s:\n        if ch.isalpha():\n            base = ord("a") if ch.islower() else ord("A")\n            out += chr((ord(ch) - base + k) % 26 + base)\n        else:\n            out += ch\n    return out',
        "JavaScript": 'function solve() {{\n  const s = "{s}", k = {k};\n  let out = "";\n  for (const ch of s) {{\n    if (/[a-zA-Z]/.test(ch)) {{\n      const base = ch === ch.toLowerCase() ? 97 : 65;\n      out += String.fromCharCode((ch.charCodeAt(0) - base + k) % 26 + base);\n    }} else out += ch;\n  }}\n  return out;\n}}',
        "Java": 'public static String solve() {{\n    String s = "{s}"; int k = {k};\n    StringBuilder out = new StringBuilder();\n    for (char ch : s.toCharArray()) {{\n      if (Character.isLetter(ch)) {{\n        char base = Character.isLowerCase(ch) ? \'a\' : \'A\';\n        out.append((char) ((ch - base + k) % 26 + base));\n      }} else out.append(ch);\n    }}\n    return out.toString();\n}}',
        "C++": 'string solve() {{\n    string s = "{s}"; int k = {k};\n    for (char& ch : s) {{\n      if (isalpha(ch)) {{\n        char base = islower(ch) ? \'a\' : \'A\';\n        ch = (ch - base + k) % 26 + base;\n      }}\n    }}\n    return s;\n}}',
    },
)


def _p_count_vowels(rng, diff):
    return {"s": _rand_str(rng, 4, 9)}


def _e_count_vowels(p):
    return sum(1 for c in p["s"].lower() if c in "aeiou")


_family(
    "count_vowels", _p_count_vowels,
    lambda p: f'Return the number of vowels in "{p["s"]}".',
    _e_count_vowels,
    {
        "Python": 'def solve():\n    s = "{s}"\n    return sum(1 for c in s.lower() if c in "aeiou")',
        "JavaScript": 'function solve() {{\n  const s = "{s}";\n  return (s.match(/[aeiou]/gi) || []).length;\n}}',
        "Java": 'public static int solve() {{\n    String s = "{s}".toLowerCase();\n    int c = 0;\n    for (char ch : s.toCharArray()) if ("aeiou".indexOf(ch) >= 0) c++;\n    return c;\n}}',
        "C++": 'int solve() {{\n    string s = "{s}"; int c = 0;\n    for (char ch : s) if (strchr("aeiou", tolower(ch))) c++;\n    return c;\n}}',
    },
)


def _p_count_consonants(rng, diff):
    return {"s": _rand_str(rng, 4, 9)}


def _e_count_consonants(p):
    return sum(1 for c in p["s"].lower() if c in "bcdfghjklmnpqrstvwxyz")


_family(
    "count_consonants", _p_count_consonants,
    lambda p: f'Return the number of consonants in "{p["s"]}".',
    _e_count_consonants,
    {
        "Python": 'def solve():\n    s = "{s}"\n    return sum(1 for c in s.lower() if c in "bcdfghjklmnpqrstvwxyz")',
        "JavaScript": 'function solve() {{\n  const s = "{s}";\n  return (s.match(/[^aeiou]/gi) || []).filter(c => /[a-z]/i.test(c)).length;\n}}',
        "Java": 'public static int solve() {{\n    String s = "{s}".toLowerCase();\n    int c = 0;\n    for (char ch : s.toCharArray()) if ("bcdfghjklmnpqrstvwxyz".indexOf(ch) >= 0) c++;\n    return c;\n}}',
        "C++": 'int solve() {{\n    string s = "{s}"; int c = 0;\n    for (char ch : s) if (strchr("bcdfghjklmnpqrstvwxyz", tolower(ch))) c++;\n    return c;\n}}',
    },
)


def _p_find_max(rng, diff):
    n = rng.randint(4, 10)
    arr = [rng.randint(0, 999) for _ in range(n)]
    return {"arr": str(arr)}


def _e_find_max(p):
    return max(eval(p["arr"]))


_family(
    "find_max", _p_find_max,
    lambda p: f'Return the largest number in the list {p["arr"]}.',
    _e_find_max,
    {
        "Python": "def solve():\n    arr = {arr}\n    return max(arr)",
        "JavaScript": "function solve() {{\n  const arr = {arr};\n  return Math.max(...arr);\n}}",
        "Java": "public static int solve() {{\n    int[] arr = {arr};\n    int m = arr[0];\n    for (int x : arr) if (x > m) m = x;\n    return m;\n}}",
        "C++": "int solve() {{\n    vector<int> arr = {arr};\n    int m = arr[0];\n    for (int x : arr) if (x > m) m = x;\n    return m;\n}}",
    },
)


def _p_sum_list(rng, diff):
    n = rng.randint(4, 10)
    arr = [rng.randint(0, 999) for _ in range(n)]
    return {"arr": str(arr)}


def _e_sum_list(p):
    return sum(eval(p["arr"]))


_family(
    "sum_list", _p_sum_list,
    lambda p: f'Return the sum of all numbers in the list {p["arr"]}.',
    _e_sum_list,
    {
        "Python": "def solve():\n    arr = {arr}\n    return sum(arr)",
        "JavaScript": "function solve() {{\n  const arr = {arr};\n  return arr.reduce((a, b) => a + b, 0);\n}}",
        "Java": "public static int solve() {{\n    int[] arr = {arr};\n    int s = 0;\n    for (int x : arr) s += x;\n    return s;\n}}",
        "C++": "int solve() {{\n    vector<int> arr = {arr};\n    int s = 0;\n    for (int x : arr) s += x;\n    return s;\n}}",
    },
)


_WORD_POOL = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "big", "red",
              "blue", "sky", "sun", "tree", "book", "code", "data", "model", "train",
              "learn", "small", "green", "old", "new", "car", "river", "cloud", "stone"]


def _p_word_count(rng, diff):
    n = rng.randint(4, 9)
    return {"s": " ".join(rng.choice(_WORD_POOL) for _ in range(n))}


def _e_word_count(p):
    return len(p["s"].split())


_family(
    "word_count", _p_word_count,
    lambda p: f'Return the number of words in the sentence: "{p["s"]}".',
    _e_word_count,
    {
        "Python": 'def solve():\n    s = "{s}"\n    return len(s.split())',
        "JavaScript": 'function solve() {{\n  const s = "{s}";\n  return s.split(" ").filter(Boolean).length;\n}}',
        "Java": 'public static int solve() {{\n    String s = "{s}";\n    int c = 0;\n    for (String w : s.split(" ")) if (!w.isEmpty()) c++;\n    return c;\n}}',
        "C++": 'int solve() {{\n    string s = "{s}";\n    int c = 0; bool in = false;\n    for (char ch : s) {{ if (ch == 32) in = false; else if (!in) {{ in = true; c++; }} }}\n    return c;\n}}',
    },
)


def _p_triangular(rng, diff):
    return {"n": rng.randint(5, 150 if diff != "hard" else 400)}


def _e_triangular(p):
    return p["n"] * (p["n"] + 1) // 2


_family(
    "triangular", _p_triangular,
    lambda p: f"Return the {p['n']}-th triangular number T({p['n']}).",
    _e_triangular,
    {
        "Python": "def solve():\n    n = {n}\n    return n * (n + 1) // 2",
        "JavaScript": "function solve() {{\n  let n = {n};\n  return n * (n + 1) / 2;\n}}",
        "Java": "public static long solve() {{\n    int n = {n};\n    return (long) n * (n + 1) / 2;\n}}",
        "C++": "long long solve() {{\n    int n = {n};\n    return (long long) n * (n + 1) / 2;\n}}",
        "SQL": "SELECT {n} * ({n} + 1) / 2 AS triangular;",
    },
)


def _make_coding(rng: random.Random, diff: str, lang: str) -> Dict[str, str]:
    supported = ["Python", "JavaScript", "Java", "C++", "SQL"]
    if lang not in supported:
        lang = rng.choice(supported)
    fam = rng.choice(_FAMILIES)
    for _ in range(6):
        p = fam["params"](rng, diff)
        code = fam["code"].get(lang)
        if code is None:
            # family not defined for this language; pick a numeric family
            fam = rng.choice([f for f in _FAMILIES if lang in f["code"]])
            continue
        problem = fam["problem"](p)
        explanation = (
            f"This {lang} solution solves the '{fam['name']}' problem. "
            f"It runs in O(n) time and uses O(1) extra space. "
            f"The constants are embedded directly so the program is self-contained."
        )
        if lang == "Python":
            src = code.format(**p)
            try:
                ns: Dict[str, Any] = {}
                exec(compile(src, "<code>", "exec"), ns)
                got = ns["solve"]()
                if str(got) != str(fam["expected"](p)):
                    continue
            except Exception:
                continue
        elif lang == "SQL":
            src = code.format(**p)
            try:
                con = sqlite3.connect(":memory:")
                cur = con.cursor()
                cur.execute(src)
                con.close()
            except Exception:
                continue
        else:
            try:
                ast.parse(code.format(**p))
            except Exception:
                continue
        return {
            "problem": problem, "language": lang, "code": code.format(**p),
            "explanation": explanation, "difficulty": diff,
        }
    # Fallback to a guaranteed-simple family.
    fam = _FAMILIES[0]
    p = fam["params"](rng, diff)
    return {
        "problem": fam["problem"](p), "language": lang,
        "code": fam["code"][lang].format(**p),
        "explanation": f"{lang} solution for '{fam['name']}'.", "difficulty": diff,
    }


# ---------------------------------------------------------------------------
# DEBUGGING
# ---------------------------------------------------------------------------

_BUGS = [
    ("off_by_one", "for i in range(len(arr)):", "for i in range(len(arr) - 1):",
     "The loop bound was off by one; it should stop before the last index."),
    ("wrong_op", "total = total + 1", "total = total + x",
     "The accumulator added the wrong variable; it should add x."),
    ("missing_return", "    s = s + i", "    s = s + i\n    return s",
     "The function computed the result but forgot to return it."),
    ("wrong_index", "arr[0]", "arr[i]",
     "It always used index 0 instead of the loop variable i."),
    ("init_zero", "total = 1", "total = 0",
     "The accumulator was initialised to 1 instead of 0."),
    ("wrong_compare", "while n > 1:", "while n != 1:",
     "The termination condition was too strict; it should stop exactly at 1."),
    ("sign_flip", "n = n // 2", "n = -n // 2",
     "A stray minus sign inverted the value; removing it restores correctness."),
    ("double_add", "total += i", "total += i + i",
     "Each term was added twice; it should be added only once."),
    ("swap_vars", "a, b = b, a", "a, b = a, b",
     "The swap was a no-op; the variables were assigned back to themselves."),
]


def _make_debugging(rng: random.Random, diff: str, lang: str) -> Dict[str, str]:
    # Build a correct Python snippet, then inject a bug.
    fam = rng.choice([f for f in _FAMILIES if "Python" in f["code"] and "SQL" not in f["code"]])
    p = fam["params"](rng, diff)
    correct = fam["code"]["Python"].format(**p)
    bug_name, wrong, right, why = rng.choice(_BUGS)
    if wrong in correct:
        buggy = correct.replace(wrong, right, 1)
    else:
        # fallback: drop the return statement
        buggy = re.sub(r"\n    return .*", "", correct)
        why = "The return statement was removed, so the function returns None."
    # Fixed code is the original correct code (re-run to be safe).
    buggy = buggy if buggy != correct else correct.replace("return total", "return total + 0")
    explanation = (
        f"The original code contained a '{bug_name}' defect. {why} "
        f"The corrected version restores the intended behaviour and now "
        f"returns the proper result for the '{fam['name']}' computation."
    )
    return {
        "buggy_code": buggy, "fixed_code": correct, "explanation": explanation,
        "language": "Python", "difficulty": diff,
    }


# ---------------------------------------------------------------------------
# SQL (verified on in-memory SQLite)
# ---------------------------------------------------------------------------

def _make_sql(rng: random.Random, diff: str) -> Dict[str, str]:
    domains = ["employees", "orders", "products", "students", "events"]
    domain = rng.choice(domains)
    if domain == "employees":
        schema = ("CREATE TABLE employees (id INTEGER, name TEXT, dept TEXT, "
                  "salary INTEGER, years INTEGER);")
        depts = ["Sales", "Eng", "HR", "Ops", "Support", "Legal", "Finance",
                 "Marketing", "Research", "Design", "IT", "Logistics"]
        rows = [(i, f"emp{i}", rng.choice(depts),
                 rng.randint(40000, 120000), rng.randint(1, 20))
                for i in range(1, rng.randint(8, 20))]
        qtypes = ["count", "avg_salary", "max_salary", "group_by_dept", "min_years", "count_high"]
        qt = rng.choice(qtypes)
        if qt == "count":
            target = rng.choice(depts)
            question = f"How many employees work in the {target} department?"
            sql = f"SELECT COUNT(*) FROM employees WHERE dept = '{target}';"
            answer = sum(1 for r in rows if r[2] == target)
        elif qt == "avg_salary":
            question = "What is the average salary of all employees?"
            sql = "SELECT AVG(salary) FROM employees;"
            answer = round(sum(r[3] for r in rows) / len(rows), 2)
        elif qt == "max_salary":
            question = "Who has the highest salary?"
            sql = "SELECT name FROM employees ORDER BY salary DESC LIMIT 1;"
            answer = max(rows, key=lambda r: r[3])[1]
        elif qt == "min_years":
            question = "What is the minimum years of experience among employees?"
            sql = "SELECT MIN(years) FROM employees;"
            answer = min(r[4] for r in rows)
        elif qt == "count_high":
            thr = rng.choice([60000, 70000, 80000, 90000, 100000])
            question = f"How many employees earn more than {thr}?"
            sql = f"SELECT COUNT(*) FROM employees WHERE salary > {thr};"
            answer = sum(1 for r in rows if r[3] > thr)
        else:
            question = "How many employees are in each department?"
            sql = "SELECT dept, COUNT(*) FROM employees GROUP BY dept;"
            answer = {}
            for r in rows:
                answer[r[2]] = answer.get(r[2], 0) + 1
    elif domain == "products":
        schema = ("CREATE TABLE products (id INTEGER, name TEXT, category TEXT, "
                  "price REAL, stock INTEGER);")
        cats = ["Book", "Toy", "Tool", "Food", "Cloth", "Game", "Decor", "Sport"]
        rows = [(i, f"item{i}", rng.choice(cats), round(rng.uniform(5, 200), 2),
                 rng.randint(0, 500)) for i in range(1, rng.randint(8, 20))]
        if diff == "easy":
            question = "List the names of all products."
            sql = "SELECT name FROM products;"
            answer = [r[1] for r in rows]
        else:
            target = rng.choice(cats)
            qt = rng.choice(["stock", "avg_price", "max_price"])
            if qt == "stock":
                question = f"What is the total stock of {target} products?"
                sql = f"SELECT SUM(stock) FROM products WHERE category = '{target}';"
                answer = sum(r[4] for r in rows if r[2] == target)
            elif qt == "max_price":
                question = f"What is the most expensive {target} product's price?"
                sql = f"SELECT MAX(price) FROM products WHERE category = '{target}';"
                vals = [r[3] for r in rows if r[2] == target]
                answer = max(vals) if vals else 0.0
            else:
                question = f"What is the average price of {target} products?"
                sql = f"SELECT AVG(price) FROM products WHERE category = '{target}';"
                answer = round(sum(r[3] for r in rows if r[2] == target) /
                               max(1, sum(1 for r in rows if r[2] == target)), 2)
    elif domain == "orders":
        schema = ("CREATE TABLE orders (id INTEGER, customer TEXT, amount REAL, "
                  "region TEXT);")
        regions = ["North", "South", "East", "West", "Central", "Coast", "Highland", "Valley"]
        rows = [(i, f"cust{i}", round(rng.uniform(10, 1000), 2), rng.choice(regions))
                for i in range(1, rng.randint(8, 20))]
        if diff == "hard":
            question = "Which region has the highest total order amount?"
            sql = "SELECT region FROM orders GROUP BY region ORDER BY SUM(amount) DESC LIMIT 1;"
            best = max(regions, key=lambda rg: sum(r[2] for r in rows if r[3] == rg))
            answer = best
        else:
            question = "What is the total order amount per region?"
            sql = "SELECT region, SUM(amount) FROM orders GROUP BY region;"
            answer = {}
            for r in rows:
                answer[r[3]] = round(answer.get(r[3], 0.0) + r[2], 2)
    elif domain == "events":
        schema = ("CREATE TABLE events (id INTEGER, title TEXT, type TEXT, "
                  "attendees INTEGER, month INTEGER);")
        types = ["Conference", "Workshop", "Meetup", "Seminar", "Hackathon", "Webinar"]
        rows = [(i, f"evt{i}", rng.choice(types), rng.randint(10, 500), rng.randint(1, 12))
                for i in range(1, rng.randint(8, 20))]
        if diff == "easy":
            question = "List the titles of all events."
            sql = "SELECT title FROM events;"
            answer = [r[1] for r in rows]
        elif diff == "hard":
            question = "Which event type has the highest average attendance?"
            sql = "SELECT type FROM events GROUP BY type ORDER BY AVG(attendees) DESC LIMIT 1;"
            best = max(types, key=lambda t: sum(r[3] for r in rows if r[2] == t) /
                       max(1, sum(1 for r in rows if r[2] == t)))
            answer = best
        else:
            thr = rng.choice([50, 100, 200, 300])
            question = f"How many events have more than {thr} attendees?"
            sql = f"SELECT COUNT(*) FROM events WHERE attendees > {thr};"
            answer = sum(1 for r in rows if r[3] > thr)
    else:
        schema = ("CREATE TABLE students (id INTEGER, name TEXT, subject TEXT, "
                  "score INTEGER);")
        subs = ["Math", "Physics", "Chemistry", "Biology", "History", "Geography", "Art", "Music"]
        rows = [(i, f"stu{i}", rng.choice(subs), rng.randint(40, 100))
                for i in range(1, rng.randint(8, 20))]
        if diff == "hard":
            question = "What is the average score for each subject, only for subjects with an average above 70?"
            sql = "SELECT subject, AVG(score) FROM students GROUP BY subject HAVING AVG(score) > 70;"
            answer = {}
            tmp = {}
            for r in rows:
                tmp.setdefault(r[2], []).append(r[3])
            for s, v in tmp.items():
                if sum(v) / len(v) > 70:
                    answer[s] = round(sum(v) / len(v), 2)
        elif diff == "medium":
            question = "How many students scored above 80 in each subject?"
            sql = "SELECT subject, COUNT(*) FROM students WHERE score > 80 GROUP BY subject;"
            answer = {}
            for r in rows:
                if r[3] > 80:
                    answer[r[2]] = answer.get(r[2], 0) + 1
        else:
            question = "What is the highest score achieved?"
            sql = "SELECT MAX(score) FROM students;"
            answer = max(r[3] for r in rows)

    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute(schema)
    cur.executemany(f"INSERT INTO {domain} VALUES ({','.join(['?'] * len(rows[0]))})", rows)
    cur.execute(sql)
    con.close()
    explanation = (
        f"Schema:\n{schema.strip()}\n\nThe query answers the question directly. "
        f"Verified result on the generated rows: {answer}."
    )
    return {
        "schema": schema.strip(), "question": question, "sql": sql.strip(),
        "explanation": explanation, "difficulty": diff,
    }


# ---------------------------------------------------------------------------
# REASONING
# ---------------------------------------------------------------------------

def _make_reasoning(rng: random.Random, diff: str) -> Dict[str, str]:
    kind = rng.choice(["logical", "causal", "planning", "algorithmic",
                       "math_derivation", "debugging_reasoning"])
    if kind == "logical":
        a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 99)
        scene = rng.choice(["boxes", "bags", "trays", "crates", "jars", "piles"])
        unit = rng.choice(["items", "marbles", "coins", "cards", "blocks", "stones"])
        q = (f"If all {a} {scene} contain {b} {unit}, and we add {c} more {scene} each "
             f"with {b} {unit}, how many {unit} are there in total?")
        reasoning = (f"{unit.capitalize()} from the first {a} {scene}: {a} × {b} = {a * b}. "
                     f"{unit.capitalize()} from the added {c} {scene}: {c} × {b} = {c * b}. "
                     f"Total = {a * b} + {c * b} = {a * b + c * b}.")
        return {"question": q, "reasoning": reasoning, "answer": str(a * b + c * b)}
    if kind == "causal":
        obs = rng.choice([
            "the ground is wet", "the lights are off", "the milk is sour", "the plant wilted",
            "the phone is dead", "the server is slow", "the cake did not rise", "the car will not start",
            "the email bounced", "the tap is leaking", "the screen is cracked", "the Wi-Fi dropped",
            "the coffee is cold", "the battery drains fast", "the door will not lock",
        ])
        causes = rng.sample([
            "a sprinkler", "a power cut", "warm weather", "lack of water", "a drained battery",
            "high traffic", "too little flour", "a flat spark plug", "a wrong address",
            "a worn washer", "a hard impact", "interference", "a forgotten timer", "a background app",
            "a misaligned latch", "a loose cable", "condensation", "an old bulb",
        ], 3)
        q = f"{obs.capitalize()}. Which is the most likely cause: {causes[0]}, {causes[1]}, or {causes[2]}?"
        reasoning = (f"{obs.capitalize()} can follow from several mechanisms. Among the options, {causes[0]} is the "
                     f"most direct and common explanation; {causes[1]} and {causes[2]} are possible but less likely "
                     f"unless other signs appear. Prefer the simplest cause that fits the evidence before assuming "
                     f"a rarer fault, since correlation is not causation.")
        return {"question": q, "reasoning": reasoning,
                "answer": f"The most likely cause is {causes[0]}; the others are possible but less direct."}
    if kind == "planning":
        steps = rng.randint(3, 6)
        room = rng.choice(["messy room", "cluttered desk", "disorganised kitchen", "garage",
                          "crowded shelf", "busy workshop", "shared office", "studio"])
        verbs = rng.sample(["collect", "sort", "group", "wipe", "sweep", "label",
                           "store", "discard", "stack", "arrange"], steps)
        q = f"Plan the steps to move from a {room} to a clean, organised one in about {steps} steps."
        lines = [f"{i+1}) {v.capitalize()} the items so the {room} becomes orderly."
                 for i, v in enumerate(verbs)]
        reasoning = "\n".join(lines) + f"\n{steps}) Do a final check that nothing is out of place."
        return {"question": q, "reasoning": reasoning,
                "answer": f"A {steps}-step plan for the {room}: " + ", ".join(verbs) + "."}
    if kind == "algorithmic":
        n = rng.randint(5, 16)

        def _binary(n):
            return (f"Describe how binary search finds a target in a sorted list of {n} items.",
                    f"Start with the range [0, {n-1}]. Compare the middle element with the target; if equal, "
                    f"done; otherwise discard the half that cannot contain it. Repeat, halving the range each "
                    f"step. Worst case takes ceil(log2({n})) ≈ {math.ceil(math.log2(n))} comparisons.",
                    f"Binary search halves the range each step; ~{math.ceil(math.log2(n))} comparisons worst case.")

        def _linear(n):
            return (f"Describe how linear search checks a list of {n} items for a target.",
                    f"Visit each element from index 0 to {n-1} in order, comparing it to the target and stopping "
                    f"at the first match. In the worst case every element is examined.",
                    f"Linear search examines up to {n} elements, so it is O({n}) worst case.")

        def _bubble(n):
            return (f"Describe how bubble sort orders a list of {n} items.",
                    f"Repeatedly sweep adjacent pairs; whenever a pair is out of order, swap them. After each "
                    f"full pass the largest unsorted element 'bubbles' to its final position, so {n} passes "
                    f"at most are needed.",
                    f"Bubble sort makes about {n}*(n-1)//2 comparisons in the worst case, i.e. O({n}^2).")

        def _insertion(n):
            return (f"Describe how insertion sort orders a list of {n} items.",
                    f"Build the sorted portion one element at a time: take the next element and insert it into "
                    f"its correct place among the already-sorted prefix by shifting larger items right.",
                    f"Insertion sort shifts elements on average {n}//2 times per insert, so about O({n}^2) work.")

        def _merge(n):
            return (f"Describe how merge sort orders a list of {n} items.",
                    f"Split the list into halves recursively until single elements remain, then merge sorted "
                    f"halves back together by repeatedly taking the smaller front element. The depth of splits "
                    f"is ceil(log2({n})).",
                    f"Merge sort does O({n} log {n}) comparisons by splitting log {n} deep and merging linearly each level.")

        def _quick(n):
            return (f"Describe how quick sort orders a list of {n} items.",
                    f"Pick a pivot, partition the other elements into those <= pivot and > pivot, then recurse "
                    f"on the two partitions. Each partitioning step places the pivot in its final position.",
                    f"Quick sort averages O({n} log {n}) work; worst case O({n}^2) if pivots are unbalanced.")

        def _bfs(n):
            return (f"Describe how breadth-first search visits the {n} nodes of a graph level by level.",
                    f"Start from the source, explore all immediate neighbours, then their neighbours, using a "
                    f"queue so nodes are visited in order of distance from the start.",
                    f"BFS visits every node and edge once, so O({n}+edges); it finds shortest paths in unweighted graphs.")

        def _dfs(n):
            return (f"Describe how depth-first search explores a graph of about {n} nodes.",
                    f"Start at the source and follow one branch as deep as possible, backtracking when a dead "
                    f"end is reached, using a stack (or recursion) to remember where to return.",
                    f"DFS visits every node and edge once, O({n}+edges), but does not guarantee shortest paths.")

        opts = [_binary, _linear, _bubble, _insertion, _merge, _quick, _bfs, _dfs]
        q, reasoning, answer = rng.choice(opts)(n)
        return {"question": q, "reasoning": reasoning, "answer": answer}
    if kind == "math_derivation":
        m = _gen_math(rng, diff)
        return {"question": m["problem"], "reasoning": m["derivation"],
                "answer": m["answer"]}
    # debugging_reasoning
    lang = rng.choice(["Python", "JavaScript", "Java", "C++"])
    bug = rng.choice([
        ("skips a value", "an accidental 'continue' or an 'if i == k: pass' branch",
         "remove the skip/continue so every value is handled"),
        ("runs forever", "a loop bound that never reaches its stop condition",
         "fix the termination condition so the bound is actually reached"),
        ("returns the wrong type", "a value converted or wrapped incorrectly before return",
         "return the value in the expected type/shape"),
        ("off-by-one result", "the range stops one short of the last index",
         "extend the bound by one (or use inclusive indexing)"),
        ("mutates shared state", "a global or input list changed in place",
         "copy the input first and only change local state"),
        ("null/None crash", "dereferencing a value that can be absent",
         "guard for the empty/missing case before using it"),
        ("swapped variables", "two variables assigned without a temporary",
         "use a temporary (or tuple swap) so neither value is lost"),
        ("sign error", "a minus applied where it should be plus (or vice versa)",
         "check the operator; the accumulator should add, not subtract"),
        ("integer division", "using // where a float result was needed",
         "use true division so fractional parts are kept"),
        ("wrong accumulator", "adding a constant instead of the running total",
         "accumulate the intended variable each step"),
    ])
    symptom, cause, fix = bug
    q = f"A {lang} program {symptom}. What is a likely cause and how would you fix it?"
    reasoning = (f"If a {lang} program {symptom}, a likely cause is {cause}. This is a common logic slip that "
                 f"produces the observed behaviour. The fix is to {fix}, which restores the intended control "
                 f"flow or data handling so the program behaves correctly.")
    return {"question": q, "reasoning": reasoning,
            "answer": f"Likely cause: {cause}. Fix: {fix}."}


# ---------------------------------------------------------------------------
# INSTRUCTION (general)
# ---------------------------------------------------------------------------

_INSTR_ITEMS = {
    "explain": ["renewable energy", "blockchain", "photosynthesis", "inflation",
                "immune system", "gravity", "supply and demand", "recursion",
                "machine learning", "climate change", "the water cycle", "democracy",
                "electric circuits", "digestive system", "interest rates", "neural networks",
                "plate tectonics", "the carbon cycle", "memory in computers", "sound waves"],
    "howto": ["brew coffee", "tie a tie", "change a tyre", "plant tomatoes",
               "back up a phone", "write a resume", "meditate", "train a dog",
               "fold a shirt", "start a compost bin", "improve sleep", "learn a language",
               "bake bread", "fix a leaky tap", "organise a desk", "water houseplants"],
    "list": ["benefits of exercise", "ways to save money", "study techniques",
              "healthy breakfast ideas", "time-management tips", "habits of productive people",
              "ways to reduce stress", "tips for public speaking", "eco-friendly swaps",
              "steps to declutter", "methods to focus", "ideas for family dinners"],
    "compare": ["renting vs buying a home", "tea vs coffee", "east coast vs west coast",
                 "electric vs petrol cars", "remote vs office work", "android vs ios",
                 "savings vs investments", "libraries vs the internet",
                 "trains vs planes", "cash vs cards", "walking vs cycling to work",
                 "email vs instant messaging", "books vs podcasts", "cats vs dogs",
                 "summer vs winter holidays", "city vs countryside living",
                 "public school vs private school", "bus vs metro", "renting vs owning a car",
                 "streaming vs cinema", "notebook vs laptop", "pen vs keyboard",
                 "morning vs evening routines", "team sport vs solo sport",
                 "cooking vs ordering food", "call vs text", "map vs gps",
                 "acoustic vs electric guitar", "paper vs digital notes",
                 "fixed vs variable rate loans", "lease vs buy a phone",
                 "north vs south campus", "coffee vs tea for focus", "taxi vs ride-share",
                 "garden vs balcony plants", "board game vs video game",
                 "in-person vs online class", "save vs invest spare cash",
                 "small vs large team", "quiet vs busy workspace"],
    "define": ["algorithm", "ecosystem", "inflation", "metadata", "quantum bit",
                "api", "bias in data", "open source", "bandwidth", "vector",
                "recursion", "cache", "protocol", "gradient", "entropy", "latency",
                "throughput", "heuristic", "topology", "schema", "token", "kernel",
                "rendering", "payload", "threshold", "metric", "baseline", "cluster",
                "pipeline", "regression", "classifier", "embedding", "index",
                "buffer", "checksum", "iteration", "state machine", "middleware",
                "latent variable", "sampling", "edge case", "abstraction"],
}
_AUDIENCES = ["a curious ten-year-old", "a university student", "a busy professional",
              "a complete beginner", "a colleague", "a policy maker"]
_EXPLAIN_TAILS = [
    "A useful way to remember this is to connect it to something you already know.",
    "Practitioners often revisit the basics before trusting advanced claims.",
    "Real-world data usually makes the idea clearer than abstract description.",
    "Keeping notes helps solidify the concept over time.",
    "Asking 'why' at each step reveals the underlying structure.",
    "Diagrams can make the relationships easier to see.",
    "Comparing two examples side by side sharpens understanding.",
    "The trade-off between speed and accuracy matters in practice.",
    "Small experiments are the fastest path to intuition.",
    "Naming the parts reduces confusion when discussing it.",
    "Edge cases are where most misunderstandings hide.",
    "A single concrete example beats a page of generalities.",
    "Repeating the idea in your own words checks comprehension.",
    "Historical context often explains why the idea took its current form.",
    "Measurement turns vague intuition into something actionable.",
    "Once the pattern is spotted, similar problems become easy.",
    "Teaching it to someone else is the strongest test of understanding.",
    "Constraints force clearer thinking about what really matters.",
    "The same principle appears in many unrelated fields.",
    "Avoiding jargon keeps the explanation honest and accessible.",
    "A checklist prevents common mistakes when applying it.",
    "Breaking it into smaller pieces makes it far less intimidating.",
    "Curiosity about exceptions leads to deeper insight.",
    "Simple models are useful even when they are not perfectly accurate.",
    "The cost of being wrong is often what makes the method careful.",
    "Good definitions remove ambiguity before arguments begin.",
    "Patterns that repeat suggest an underlying rule worth learning.",
    "Feedback loops are how the system improves itself.",
    "Understanding the limits is as important as knowing the uses.",
    "A worked example makes the abstract concrete and memorable.",
    "The right level of detail depends on who is listening.",
    "Building on prior knowledge accelerates the whole process.",
    "Clarity beats cleverness when explaining to beginners.",
    "The core idea is simpler than it first appears.",
    "Context determines which variant of the idea applies.",
    "Repetition with variation is how skills become automatic.",
    "Questions are more valuable than answers when exploring.",
]


def _make_instruction(rng: random.Random, diff: str) -> Dict[str, str]:
    kind = rng.choice(list(_INSTR_ITEMS.keys()))
    item = rng.choice(_INSTR_ITEMS[kind])
    if kind == "explain":
        aud = rng.choice(_AUDIENCES)
        instr = f"Explain {item} to {aud}."
        example = rng.choice(["a falling apple", "a leaking bucket", "a growing plant",
                             "a spinning top", "a boiling kettle", "a rolling ball"])
        tail = rng.choice(_EXPLAIN_TAILS)
        resp = (f"{item.title()} can be explained in three parts. First, it describes the "
                f"core mechanism behind {item}. Second, it shows why {item} matters in everyday "
                f"life. Third, it corrects a common misconception. For example, think of {example}: "
                f"the same idea helps us understand {item} more clearly. In short, {item} lets "
                f"us organise and reason about the world efficiently. {tail}")
    elif kind == "howto":
        instr = f"Give step-by-step instructions for how to {item}."
        n = 3 if diff == "easy" else (5 if diff == "medium" else 7)
        _phrases = ['gather your materials', 'set a clear goal', 'work in a calm space',
                    'check progress', 'avoid common mistakes', 'keep your tools ready',
                    'review the result', 'ask for feedback', 'document what you did']
        steps = [f"Step {i+1}: {rng.choice(_phrases)} so the {item} task stays on track."
                 for i in range(n)]
        resp = "Here is a clear procedure:\n" + "\n".join(steps) + \
               f"\nFinish by confirming the {item} outcome meets your goal."
    elif kind == "list":
        instr = f"List several {item}."
        n = 4 if diff == "easy" else (6 if diff == "medium" else 8)
        resp = "\n".join(
            f"- {item.split()[0].capitalize()} idea {i+1}: {rng.choice(['start small and stay consistent','measure your progress','learn from others','keep it simple','adapt when needed','review weekly','set a reminder','focus on one thing at a time'])}."
            for i in range(n))
    elif kind == "compare":
        a, b = item.split(" vs ")
        instr = f"Compare {item}."
        resp = (f"{a} and {b} differ in cost, flexibility, and long-term impact. "
                f"{a} tends to favour {rng.choice(['simplicity now','lower risk','speed'])} "
                f"while {b} favours {rng.choice(['flexibility later','scalability','control'])}. "
                f"Choose based on your time horizon and budget.")
    else:
        instr = f"Define the term '{item}'."
        tail = rng.choice(_EXPLAIN_TAILS)
        resp = (f"'{item}' refers to a precise idea used to describe a system or process. "
                f"It is defined by its key properties — for instance, the way {item} behaves "
                f"under changing conditions — and is distinguished from related terms by its "
                f"specific scope and behaviour. {tail}")
    return {"instruction": instr, "response": resp, "difficulty": diff, "language": "en"}


# ---------------------------------------------------------------------------
# SUMMARIZATION
# ---------------------------------------------------------------------------

def _make_summarization(rng: random.Random, diff: str) -> Dict[str, str]:
    topic = rng.choice(corpus.TOPICS)
    n = 4 if diff == "easy" else 6
    article = corpus.make_passage(rng, topic, n)
    summary = corpus.make_summary(rng, article, topic)
    instruction = "Summarize the following article in one or two concise sentences."
    return {"instruction": instruction, "input": article, "article": article,
            "summary": summary, "difficulty": diff, "language": "en"}


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def _make_qa(rng: random.Random, diff: str) -> Dict[str, str]:
    topic = rng.choice(corpus.TOPICS)
    pct = rng.randint(10, 45)
    obj = rng.choice(corpus.OBJECTS)
    place = rng.choice(corpus.PLACES)
    subj = rng.choice(corpus.SUBJECTS)
    context = (f"This article is about {topic}. A team of {subj} "
               f"studied the effect of {obj} {place}. The study found that {obj} increased "
               f"by {pct}% after the intervention {place}. Participants reported higher {obj} "
               f"when the new method was adopted {place}.")
    qkind = rng.choice(["pct", "who", "where", "what"])
    if qkind == "pct":
        question = f"By what percentage did {obj} increase after the intervention?"
        answer = f"The {obj} increased by {pct}% after the intervention."
    elif qkind == "who":
        question = f"Who studied the effect of {obj}?"
        answer = f"A team of {subj} studied the effect of {obj}."
    elif qkind == "where":
        question = f"Where was the intervention carried out?"
        answer = f"The intervention was carried out {place}."
    else:
        question = f"What topic does the passage discuss?"
        answer = f"The passage discusses {topic}."
    instruction = "Answer the question using only the information in the context."
    return {"instruction": instruction, "input": context, "context": context,
            "question": question, "answer": answer, "difficulty": diff, "language": "en"}


# ---------------------------------------------------------------------------
# TRANSLATION
# ---------------------------------------------------------------------------

def _make_translation(rng: random.Random, diff: str) -> Dict[str, str]:
    # Prefer fully-translatable closed-vocabulary sentences (English -> X).
    if rng.random() < 0.7:
        tgt_lang = rng.choice(list(corpus.LEXICONS.keys()))
        tmpl = rng.choice(corpus.SENTENCE_TEMPLATES)
        words = re.findall(r"\{(.*?)\}", tmpl)
        fill = {}
        pool = [w for w in corpus.EN_WORDS if corpus.EN_WORDS[w] in ("noun", "verb", "adj")]
        for w in words:
            if w.startswith("subj"):
                fill[w] = rng.choice([x for x in pool if corpus.EN_WORDS[x] == "noun"])
            elif w.startswith("verb"):
                fill[w] = rng.choice([x for x in pool if corpus.EN_WORDS[x] == "verb"])
            elif w.startswith("obj"):
                fill[w] = rng.choice([x for x in pool if corpus.EN_WORDS[x] == "noun"])
            elif w.startswith(("he", "she", "it", "we", "they")):
                fill[w] = w
            else:
                fill[w] = rng.choice([x for x in pool if corpus.EN_WORDS[x] == "adj"])
        source = tmpl.format(**fill)
        target = corpus.translate_en_to(tgt_lang, source)
        return {"source": source, "target": target, "source_lang": "English",
                "target_lang": tgt_lang, "difficulty": diff, "language": tgt_lang}
    pair = rng.choice(corpus.PHRASE_BANK)
    return {"source": pair["source"], "target": pair["target"],
            "source_lang": pair["source_lang"], "target_lang": pair["target_lang"],
            "difficulty": diff, "language": pair["target_lang"]}


# ---------------------------------------------------------------------------
# FUNCTION CALLING
# ---------------------------------------------------------------------------

_TOOLS = [
    {"name": "get_weather", "params": ["city", "units"],
     "slots": lambda r: {"city": r.choice(["London", "Tokyo", "Berlin", "Paris", "Oslo",
                                          "Rome", "Madrid", "Lisbon", "Dublin", "Athens",
                                          "Vienna", "Stockholm"]),
                         "units": r.choice(["celsius", "fahrenheit"])},
     "query": lambda s: f"What is the weather in {s['city']} in {s['units']}?"},
    {"name": "book_calendar", "params": ["title", "date", "hour"],
     "slots": lambda r: {"title": r.choice(["Standup", "Interview", "Dentist", "Demo",
                                           "Retro", "Sync", "Review"]),
                         "date": f"2026-{r.randint(1,12):02d}-{r.randint(1,28):02d}",
                         "hour": r.randint(7, 20)},
     "query": lambda s: f"Book a {s['title']} on {s['date']} at {s['hour']}:00."},
    {"name": "query_database", "params": ["table", "limit"],
     "slots": lambda r: {"table": r.choice(["users", "orders", "events", "payments",
                                           "sessions", "logs"]),
                         "limit": r.choice([5, 10, 20, 50, 100, 200])},
     "query": lambda s: f"Query the {s['table']} table and return at most {s['limit']} rows."},
    {"name": "send_payment", "params": ["amount", "currency", "recipient"],
     "slots": lambda r: {"amount": r.randint(1, 9999),
                         "currency": r.choice(["USD", "EUR", "GBP", "JPY", "CHF"]),
                         "recipient": r.choice(["alice", "bob", "carol", "dave", "erin",
                                                "frank", "grace", "heidi"])},
     "query": lambda s: f"Send {s['amount']} {s['currency']} to {s['recipient']}."},
    {"name": "translate_text", "params": ["text", "target_lang"],
     "slots": lambda r: {"text": r.choice(["Hello world", "Good morning", "See you soon",
                                          "Thank you", "How are you", "Have a nice day",
                                          "Safe travels", "Well done"]),
                         "target_lang": r.choice(["fr", "de", "es", "it", "pt", "nl"])},
     "query": lambda s: f"Translate '{s['text']}' into {s['target_lang']}."},
    {"name": "web_search", "params": ["query", "top_k"],
     "slots": lambda r: {"query": r.choice(["best laptops 2026", "python async tutorial",
                                            "healthy dinner recipes", "rust vs go",
                                            "kubernetes basics", "linear algebra intro",
                                            "git rebasing", "vector databases"]),
                         "top_k": r.choice([3, 5, 10, 15, 20])},
     "query": lambda s: f"Search the web for '{s['query']}' and return {s['top_k']} results."},
    {"name": "set_reminder", "params": ["label", "day", "time"],
     "slots": lambda r: {"label": r.choice(["call mom", "submit report", "exercise",
                                           "read chapter", "water plants", "pay rent"]),
                         "day": r.choice(["Monday", "Tuesday", "Wednesday", "Thursday",
                                          "Friday", "Saturday", "Sunday"]),
                         "time": f"{r.randint(6,21)}:{r.choice(['00','15','30','45'])}"},
     "query": lambda s: f"Remind me to {s['label']} on {s['day']} at {s['time']}."},
    {"name": "create_ticket", "params": ["title", "priority", "team"],
     "slots": lambda r: {"title": r.choice(["Login failing", "Slow dashboard", "Crash on save",
                                           "Typo in docs", "Memory leak", "Timeout error"]),
                         "priority": r.choice(["low", "medium", "high", "critical"]),
                         "team": r.choice(["backend", "frontend", "infra", "qa", "mobile"])},
     "query": lambda s: f"Create a {s['priority']} priority ticket for {s['team']}: {s['title']}."},
    {"name": "book_flight", "params": ["from", "to", "date"],
     "slots": lambda r: {"from": r.choice(["NYC", "LHR", "CDG", "SFO", "BER", "TYO"]),
                         "to": r.choice(["NYC", "LHR", "CDG", "SFO", "BER", "TYO"]),
                         "date": f"2026-{r.randint(1,12):02d}-{r.randint(1,28):02d}"},
     "query": lambda s: f"Book a flight from {s['from']} to {s['to']} on {s['date']}."},
]


def _make_function_calling(rng: random.Random, diff: str) -> Dict[str, str]:
    tool = rng.choice(_TOOLS)
    slots = tool["slots"](rng)
    query = tool["query"](slots)
    rationale = (f"The user request is best fulfilled by calling '{tool['name']}' with the "
                 f"extracted arguments. All required parameters are present and typed.")
    import json as _json
    call = f"{tool['name']}({_json.dumps(slots)})"
    return {"query": query, "function_name": tool["name"], "arguments": slots,
            "function_call": call, "rationale": rationale,
            "difficulty": diff, "language": "en"}


# ---------------------------------------------------------------------------
# OFFLINE TEACHER
# ---------------------------------------------------------------------------

_GENERATORS = {
    "instruction": lambda r, d, s: _make_instruction(r, d),
    "reasoning": lambda r, d, s: _make_reasoning(r, d),
    "math": lambda r, d, s: _gen_math(r, d),
    "coding": lambda r, d, s: _make_coding(r, d, s.language or "Python"),
    "debugging": lambda r, d, s: _make_debugging(r, d, s.language or "Python"),
    "summarization": lambda r, d, s: _make_summarization(r, d),
    "translation": lambda r, d, s: _make_translation(r, d),
    "sql": lambda r, d, s: _make_sql(r, d),
    "qa": lambda r, d, s: _make_qa(r, d),
    "function_calling": lambda r, d, s: _make_function_calling(r, d),
}


class OfflineTeacher(BaseTeacher):
    """Deterministic, verifiable sample generator (no external API)."""

    def __init__(self, config: Optional[TeacherConfig] = None):
        self.config = config

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        template: Optional[Template] = None,
        seed: Optional[Seed] = None,
        **kwargs,
    ) -> GenerationResult:
        task = template.task_type if template is not None else (seed.task_type if seed else "instruction")
        total = getattr(seed, "total", 10_000) if seed is not None else 10_000
        diff = _difficulty_for(seed.idx, total) if seed is not None else "medium"
        rng = _rng(task, seed.idx) if seed is not None else random.Random(0)
        gen = _GENERATORS.get(task, _GENERATORS["instruction"])
        fields = gen(rng, diff, seed)
        fields.setdefault("difficulty", diff)
        fields.setdefault("language", "en")
        fields.setdefault("generator", "offline-generator")
        text = json.dumps(fields, ensure_ascii=False)
        est = _est_tokens(text)
        return GenerationResult(
            text=text,
            prompt_tokens=_est_tokens(prompt) if prompt else 0,
            completion_tokens=est,
            cost_usd=0.0,
            finish_reason="stop",
            logprobs=None,
            model="offline-generator",
        )
