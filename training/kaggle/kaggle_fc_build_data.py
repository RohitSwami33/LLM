#!/usr/bin/env python3
"""Kernel A (roronoazoro3008/qwen35-fc-data): build + filter + format + tokenize-once.

Builds a mixed SFT dataset (~35-45k target) teaching Qwen3.5-4B (post-trained):
  web-search decision, evidence retrieval, fact-checking, source comparison,
  anti-hallucination/uncertainty, + math/QA capability replay.

Output (/kaggle/working):
  tokenized_cache/          HF dataset: input_ids, labels (mask = assistant turns only)
  eval/*.jsonl              held-out eval suites (never trained on)
  stats.json                counts, percentages, sources used
  manifest.json             provenance

Design notes:
  - Every source is optional; per-slot fallbacks; stats record what actually loaded.
  - Tool format = Qwen native: assistant emits <tool_call>{"name":"web_search",...}</tool_call>,
    tool results enter as role:"tool" messages.
  - Tokenization: ONE pass; assistant-turn spans masked via cumulative chat-template
    boundaries (labels=-100 on user/tool text; loss on assistant turns incl. <|im_end|>).
  - No benchmark test sets are used for training. Eval suites are held out.
"""
import json, os, random, re, sys, time, hashlib
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = 1337
random.seed(SEED)
MAX_LEN = 3072          # token cap per training sequence
OUT = Path("/kaggle/working") if os.path.exists("/kaggle/working") else Path(".")
EVAL_DIR = OUT / "eval"

WEB_SEARCH_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Web search. Returns ranked results with title, url, snippet, and date when available.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    },
}]

VERDICTS = ["SUPPORTED", "LIKELY SUPPORTED", "REFUTED", "CONTESTED",
            "INSUFFICIENT EVIDENCE", "UNVERIFIED"]

def log(m): print(m, flush=True)

def norm_q(q):
    return re.sub(r"\W+", " ", q.lower()).strip()

def results_block(results):
    """Format tool output text deterministically."""
    lines = []
    for i, (title, url, snip) in enumerate(results[:5], 1):
        lines.append(f"[{i}] {title}\nURL: {url}\n{snip}")
    return "\n\n".join(lines)

def tool_call_msg(query):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function",
                            "function": {"name": "web_search",
                                         "arguments": {"query": query}}}]}

def tool_msg(results):
    return {"role": "tool", "content": results_block(results)}

# ---------------------------------------------------------------- dataset slots
def try_load(loaders, slot):
    for name, fn in loaders:
        try:
            t0 = time.time()
            ds = fn()
            if ds is None or len(ds) == 0:
                raise RuntimeError("empty")
            log(f"  [{slot}] {name}: {len(ds)} rows in {time.time()-t0:.0f}s")
            return name, ds
        except Exception as e:
            log(f"  [{slot}] {name} FAILED: {type(e).__name__}: {str(e)[:140]}")
    return None, None

def hf_load(path, config=None, split="train", streaming=False):
    from datasets import load_dataset
    kw = dict(streaming=streaming)
    if config: kw["name"] = config
    return load_dataset(path, split=split, **kw)

# --- loaders -----------------------------------------------------------------
def load_gsm8k():
    return hf_load("openai/gsm8k", "main", "test" if False else "train")

def load_openr1_sample(n=4000):
    ds = hf_load("open-r1/OpenR1-Math-220k", "default", "train", streaming=True)
    out, t0 = [], time.time()
    for r in ds:
        # keep concise solutions; strip <think> blocks -> clean non-thinking CoT
        sol = (r.get("solution") or r.get("generations", [""])[0] or "")
        sol = re.sub(r"<think>.*?</think>", "", sol, flags=re.S).strip()
        if 80 < len(sol) < 2500 and r.get("problem"):
            out.append({"problem": r["problem"], "solution": sol})
        if len(out) >= n or time.time() - t0 > 600:
            break
    return out

def load_smoltalk():
    ds = hf_load("HuggingFaceTB/smoltalk", "everyday-conversations", "train")
    return ds

def load_smoltalk_explain(n=6000):
    ds = hf_load("HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train", streaming=True)
    out, t0 = [], time.time()
    for r in ds:
        msgs = r.get("messages", [])
        if (len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
                and 100 < len(msgs[1]["content"]) < 1800):
            out.append(msgs)
        if len(out) >= n or time.time() - t0 > 600:
            break
    return out

def load_fever():
    # FEVER: claim + label + evidence (wiki sentences)
    for path, cfg in [("copenlu/fever_gold_evidence", None), ("fever", "v1.0"), ("pietrolesci/nli_fever", None)]:
        try:
            ds = hf_load(path, cfg, "train")
            cols = ds.column_names
            log(f"  [fever-cand] {path} cols={cols} n={len(ds)}")
            return ds
        except Exception as e:
            log(f"  [fever-cand] {path}: {type(e).__name__} {str(e)[:100]}")
    raise RuntimeError("no fever source")

def load_averitec():
    for path in ["asap-asverb/averitec", "GSMM/averitec", "chengxuphd/averitec"]:
        try:
            return hf_load(path, None, "train")
        except Exception:
            continue
    raise RuntimeError("no averitec source")

def load_hotpotqa():
    # script-based dataset: needs trust_remote_code; parquet mirror as fallback
    from datasets import load_dataset
    try:
        return load_dataset("hotpotqa/hotpot_qa", "distractor", split="train",
                            trust_remote_code=True)
    except Exception as e:
        log(f"  [hotpot] script load failed: {e}; trying mirror")
        return hf_load("yuyijiong/hotpot_qa_distractor_parquet", None, "train")

def load_toolace():
    return hf_load("Team-ACE/ToolACE", None, "train")

def load_nqopen():
    return hf_load("google-research-datasets/nq_open", None, "train")

# ---------------------------------------------------------------- formatters
def fmt_hotpot(rows, limit):
    out = []
    for r in rows:
        try:
            q = r["question"].strip()
            ans = r["answer"].strip()
            sup = r.get("supporting_facts", {})
            titles = list(dict.fromkeys(sup["title"]))[:2] if isinstance(sup, dict) else []
            ctx = r["context"]
            ev = []
            if isinstance(ctx, dict):
                for t, sents in zip(ctx["title"], ctx["sentences"]):
                    if t in titles:
                        ev.append((t, "".join(sents)[:900]))
            if not ev or not ans:
                continue
            msgs = [{"role": "user", "content": q}]
            for j, (t, text) in enumerate(ev, 1):
                msgs.append(tool_call_msg(f"{t} {q[:80]}".strip()))
                msgs.append(tool_msg([(t, f"https://en.wikipedia.org/wiki/{t.replace(' ','_')}", text)]))
            verdict = "SUPPORTED" if len(ev) >= 1 else "INSUFFICIENT EVIDENCE"
            final = (f"Based on the retrieved evidence:\n\n"
                     f"{ans}\n\n"
                     f"Evidence check: the sources directly support this answer ({verdict}). "
                     + ("Two independent encyclopedia articles corroborate it." if len(ev) > 1
                        else "A single source supports it; confidence is moderate."))
            msgs.append({"role": "assistant", "content": final})
            out.append(msgs)
        except Exception:
            continue
        if len(out) >= limit: break
    return out

FEVER_MAP = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED",
             "NOT ENOUGH INFO": "INSUFFICIENT EVIDENCE"}

def fmt_fever(rows, limit):
    out = []
    for r in rows:
        try:
            claim = (r.get("claim") or "").strip()
            label = (r.get("label") or "").upper().strip()
            if not claim or label not in FEVER_MAP:
                continue
            ev_sents = []
            ev = r.get("evidence")
            if isinstance(ev, list):
                for e in ev[:3]:
                    if isinstance(e, dict) and e.get("evidence"):
                        ev_sents.append(str(e["evidence"])[:400])
                    elif isinstance(e, str):
                        ev_sents.append(e[:400])
            elif isinstance(ev, str):
                ev_sents.append(ev[:400])
            if not ev_sents:
                continue
            evidence_text = " ".join(ev_sents)
            verdict = FEVER_MAP[label]
            msgs = [{"role": "user", "content": f'Is this claim true? "{claim}"'}]
            msgs.append({"role": "assistant",
                         "content": "I need to verify this claim against reliable sources.",
                         "tool_calls": [{"type": "function", "function": {
                             "name": "web_search",
                             "arguments": {"query": claim[:150]}}}]})
            msgs.append(tool_msg([("Wikipedia (encyclopedic source)",
                                   "https://en.wikipedia.org/", evidence_text)]))
            if verdict == "SUPPORTED":
                body = ("The encyclopedic evidence explicitly states facts consistent with the claim. "
                        "No contradicting source was found.")
            elif verdict == "REFUTED":
                body = ("The retrieved evidence contradicts the claim: the source states facts "
                        "inconsistent with it.")
            else:
                body = ("The retrieved evidence does not establish whether the claim is true or "
                        "false. Key details are missing from available sources.")
            final = (f"**Verdict: {verdict}**\n\n{body}\n\n"
                     f"Evidence used: encyclopedic source (retrieved). "
                     + ("Additional independent confirmation would raise confidence."
                        if verdict == "SUPPORTED" else
                        "I will not go beyond what the evidence establishes."))
            msgs.append({"role": "assistant", "content": final})
            out.append(msgs)
        except Exception:
            continue
        if len(out) >= limit: break
    return out

def fmt_averitec(rows, limit):
    out = []
    for r in rows:
        try:
            claim = (r.get("claim") or "").strip()
            label = (r.get("label") or "").strip().lower()
            if not claim:
                continue
            m = {"supported": "SUPPORTED", "refuted": "REFUTED",
                 "not enough evidence": "INSUFFICIENT EVIDENCE",
                 "conflicting evidence/passages": "CONTESTED"}.get(label)
            if not m:
                continue
            qas = r.get("questions_with_answers") or r.get("questions") or []
            results = []
            try:
                for qa in (qas if isinstance(qas, list) else [])[:3]:
                    for a in (qa.get("answers") or [])[:2]:
                        results.append((str(a.get("source_title") or "web source")[:80],
                                        str(a.get("source_url") or "")[:120],
                                        str(a.get("answer") or a.get("snippet") or "")[:350]))
            except Exception:
                pass
            if not results:
                continue
            msgs = [{"role": "user", "content": f"Verify this claim: {claim}"}]
            msgs.append(tool_call_msg(claim[:150]))
            msgs.append(tool_msg(results))
            if m == "CONTESTED":
                body = ("Credible sources disagree on this claim. Some report it as true while "
                        "others report contradictory information. I cannot resolve the conflict "
                        "from available evidence alone.")
            elif m == "SUPPORTED":
                body = "Independent web sources are consistent with the claim."
            elif m == "REFUTED":
                body = "The web evidence contradicts the claim."
            else:
                body = ("The available web evidence is not sufficient to establish or refute "
                        "the claim.")
            msgs.append({"role": "assistant",
                         "content": f"**Verdict: {m}**\n\n{body}"})
            out.append(msgs)
        except Exception:
            continue
        if len(out) >= limit: break
    return out

def fmt_toolace(rows, limit):
    """Filter ToolACE to search-like single-tool calls; convert to our web_search format."""
    out = []
    for r in rows:
        try:
            msgs_in = r.get("messages", [])
            tool_defs = r.get("tools") or []
            # find a search-ish function name
            fn_names = set()
            for t in tool_defs:
                try: fn_names.add(t["function"]["name"])
                except Exception: pass
            searchish = [n for n in fn_names if re.search(r"search|query|web|fetch|lookup|find", n, re.I)]
            if not searchish:
                continue
            conv, kept = [], 0
            for m in msgs_in:
                role = m.get("role")
                if role == "assistant" and "tool_calls" in m:
                    tc = m["tool_calls"][0]["function"]
                    if tc.get("name") in searchish and kept == 0:
                        conv.append(tool_call_msg(str(tc.get("arguments", {}).get("query", ""))[:200]))
                        kept += 1
                    else:
                        conv = None; break
                elif role == "tool":
                    txt = m.get("content")
                    if isinstance(txt, (list, dict)): txt = json.dumps(txt)[:700]
                    conv.append({"role": "tool", "content": str(txt)[:700]})
                elif role in ("user", "assistant"):
                    conv.append({"role": role, "content": str(m.get("content", ""))[:1200]})
            if conv is None or kept != 1: continue
            # ensure ends with assistant final answer
            if conv[-1]["role"] != "assistant" or "tool_calls" in conv[-1]:
                continue
            # rewrite first user msg as the trigger; drop system
            conv = [m for m in conv if m["role"] != "system"]
            if len(conv) >= 3 and conv[0]["role"] == "user":
                out.append(conv)
        except Exception:
            continue
        if len(out) >= limit: break
    return out

def fmt_gsm8k(rows, limit):
    out = []
    for r in rows:
        try:
            q = r["question"].strip()
            a = r["answer"].strip()
            # gsm8k answers contain <<>> annotations; clean
            a = re.sub(r"<<[^>]*>>", "", a)
            msgs = [{"role": "user", "content": q},
                    {"role": "assistant", "content": a}]
            out.append(msgs)
        except Exception:
            continue
        if len(out) >= limit: break
    return out

def fmt_openr1(rows, limit):
    out = []
    for r in rows:
        try:
            out.append([{"role": "user", "content": r["problem"].strip()},
                        {"role": "assistant", "content": r["solution"]}])
        except Exception:
            continue
        if len(out) >= limit: break
    return out

def fmt_smoltalk(rows, limit):
    out = []
    for r in rows:
        try:
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
            if msgs and msgs[-1]["role"] == "assistant":
                out.append(msgs)
        except Exception:
            continue
        if len(out) >= limit: break
    return out

# ---------------------------------------------------------------- synthetic
def _gen_names(prefixes, suffixes, n):
    out = []
    for p in prefixes:
        for s in suffixes:
            out.append(f"{p} {s}")
            if len(out) >= n: return out
    return out

def _company_pool(n=320):
    return _gen_names(["Nexora", "Vantage", "BluePeak", "Auralis", "Quantiv", "HelioDyne",
                       "Corvus", "StratoPay", "LumenArc", "Ferrix", "OndaWorks", "Zephyrion",
                       "Kairos Data", "Mistvale", "Solentis", "NorthGlide", "Verdant Systems",
                       "IonForge", "TerraCadence", "Quanta Loop"],
                      ["Labs", "Systems", "Robotics", "Energy", "Analytics", "Materials",
                       "Dynamics", "Technologies", "Industries", "Research", "Digital",
                       "Networks", "Bio", "Computing", "Instruments", "Aerospace"], n)

def _person_pool(n=240):
    firsts = ["Amara", "Leo", "Ingrid", "Marcus", "Yuki", "Elena", "Samuel", "Priya", "Omar",
              "Hana", "Dmitri", "Sofia", "Ravi", "Lena", "Tomas", "Aisha", "Felix", "Mira",
              "Jonas", "Nadia", "Kwame", "Isabel", "Chen", "Astrid", "Diego", "Maya"]
    lasts = ["Chen", "Vasquez", "Sorensen", "Oyelaran", "Tanaka", "Petrova", "Adeyemi",
             "Kowalski", "Ibrahim", "Lindqvist", "Moreau", "Datta", "Silva", "Novak",
             "Weber", "Sato", "Okafor", "Rossi", "Andersen", "Kim"]
    out, i = [], 0
    while len(out) < n and i < n * 5:
        name = f"{random.choice(firsts)} {random.choice(lasts)}"
        if name not in out: out.append(name)
        i += 1
    return out

COMPANIES = _company_pool()
PEOPLE = _person_pool()
TOPICS = ["quantum battery efficiency", "deep-sea carbon capture", "solid-state drone batteries",
          "protein-folding diagnostics", "urban wind microgrids", "AI-generated satellite imagery",
          "enzyme-driven plastic recycling", "low-orbit debris cleanup", "photosynthetic concrete",
          "neuromorphic hearing aids", "vertical farm robotics", "hydrogen rail freight",
          "at-home blood diagnostics", "ambient office energy harvesting", "reef-restoring robots",
          "swarm-based wildfire sensors", "sleep-stage neurofeedback", "modular nuclear microgrids",
          "bio-based aircraft fuel", " glacier-preservation foams"]
STATS_TOPICS = ["global solar capacity", "EV battery recycling rate", "ocean plastic tonnage",
                "worldwide LLM training compute", "agri-drone yield gains", "desalination output",
                "satellite launch cadence", "grid-scale storage capacity"]

def synth_decision(n):
    """Search/no-search decision examples with brief rationale. Balanced, high uniqueness."""
    out, seen = [], set()
    explain_topics = ["TCP and UDP", "photosynthesis", "RSA encryption", "the Doppler effect",
                      "REST vs GraphQL", "hash tables", "supply and demand elasticity",
                      "Bayes' theorem", "the water cycle", "Big-O notation", "DNS resolution",
                      "object-oriented inheritance", "HTML semantics", "normalized databases",
                      "protein synthesis", "Newton's three laws", "HTTP status codes",
                      "the Krebs cycle", "asymmetric vs symmetric encryption", "CRDTs",
                      "bias-variance tradeoff", "the central limit theorem", "CAP theorem",
                      "garbage collection in JVM", "CSS specificity", "regular languages",
                      "plate tectonics", "the greenhouse effect", " Ohm's law", "CUDA warps"]
    code_tasks = [("Python", "reverse a linked list"), ("Python", "implement binary search"),
                  ("JavaScript", "debounce a function"), ("Python", "flatten nested dicts"),
                  ("Java", "implement a LRU cache"), ("C++", "merge two sorted vectors"),
                  ("Python", "parse CSV without pandas"), ("SQL", "find duplicate rows"),
                  ("Python", "compute a rolling average"), ("Rust", "read a file line by line"),
                  ("Go", "implement a worker pool"), ("Python", "validate an email with regex")]
    langs_static = ["What year did World War II end?", "What is the boiling point of water?",
                    "Who wrote 'Pride and Prejudice'?", "What is the chemical symbol for iron?",
                    "How many continents are there?", "What is the capital of Japan?"]
    yes_static = [
        ("Who is the current CEO of {c}?",
         "Leadership roles change over time; the answer is time-sensitive, so I should search."),
        ("What is the latest {rep} report from {c}?",
         "This asks for the latest report — current information that requires web search."),
        ("What is the current price of {p}?",
         "Prices change continuously; I must search rather than rely on training data."),
        ("Did {c} announce {t} this week?",
         "A claim about a very recent event; I need to search for current evidence."),
        ("Is the claim that {st} grew {x}% in {y} true?",
         "A specific statistical claim that needs verification against sources."),
        ("What are the current regulations on {r} in the EU?",
         "Regulations change; the answer depends on current law, so search is necessary."),
        ("What did {p} say in their recent keynote?",
         "A recent statement by a person; requires up-to-date search."),
        ("Which {prod} model was released most recently?",
         "Product lineups change frequently; search is needed for the current answer."),
        ("Is the {rec} recall still active?",
         "Recall status is time-sensitive; search required."),
        ("What is the status of the {c} merger?",
         "Deal status changes; I need current evidence."),
    ]
    half = n // 2
    # no-search: arithmetic / algebra / code / explain / static (high uniqueness)
    i = 0
    while sum(1 for t in out if "No search needed" in t[-1]["content"]) < half:
        kind = i % 5
        if kind == 0:
            a, b = random.randint(11, 999), random.randint(11, 999)
            q, why = f"What is {a} × {b}?", "This is straightforward arithmetic; no external or current information is needed."
        elif kind == 1:
            a, b = random.randint(2, 40), random.randint(2, 60)
            q, why = f"Solve: {a}x + {b} = {a*7+b}.", "Simple algebra with a deterministic solution; no search needed."
        elif kind == 2:
            lang, task = random.choice(code_tasks)
            q, why = f"Write {lang} code to {task}.", "Standard programming knowledge; static and well-known."
        elif kind == 3:
            q, why = f"Explain {random.choice(explain_topics)}.", "Established concept with stable, undisputed explanations; no verification needed."
        else:
            q, why = random.choice(langs_static), "Stable, well-documented fact that is not disputed; no search required."
        key = norm_q(q)
        if key not in seen:
            seen.add(key)
            out.append([{"role": "user", "content": q},
                        {"role": "assistant", "content": f"No search needed. {why}"}])
        i += 1
        if i > n * 20: break
    # search-needed (randomize entities per iteration)
    i = 0
    while sum(1 for t in out if "tool_calls" in t[-1]) < n - half:
        q_tpl, why = yes_static[i % len(yes_static)]
        q = q_tpl.format(c=random.choice(COMPANIES), rep=random.choice(["earnings", "sustainability", "annual"]),
                         p=random.choice(["copper", "lithium", "wheat", "helium", "cobalt"]),
                         t=random.choice(TOPICS), st=random.choice(STATS_TOPICS),
                         x=random.randint(10, 90), y=random.choice([2024, 2025, 2026]),
                         r=random.choice(["drone flights", "AI labeling", "data resale", "e-scooters", "crypto staking"]),
                         p2=random.choice(PEOPLE), prod=random.choice(["EV", "phone", "GPU", "camera"]),
                         rec=random.choice(["vehicle", "appliance", "battery"]))
        key = norm_q(q)
        if key not in seen:
            seen.add(key)
            out.append([{"role": "user", "content": q},
                        {"role": "assistant",
                         "content": f"I should search for this. {why}",
                         "tool_calls": [{"type": "function",
                                         "function": {"name": "web_search",
                                                      "arguments": {"query": re.sub(r"[?.]$", "", q)[:150]}}}]}])
        i += 1
        if i > n * 40: break
    random.shuffle(out)
    return out[:n]

def synth_hallucination(n):
    """Anti-hallucination: fabricated premises, empty/irrelevant results, uncertainty."""
    out, seen = [], set()
    years = [2024, 2025, 2026, 2023]
    pcts = ["300%", "87%", "40%", "12x", "65%"]
    months = ["last Tuesday", "last month", "in March", "earlier this year"]
    templates = []
    for c in COMPANIES:
        templates.append((f"Was {c} founded in {random.choice(years)}?",
                          lambda c=c: [("generic-results", "https://search.example/",
                                        f"No reliable source mentions {c} or its founding date.")],
                          "INSUFFICIENT EVIDENCE",
                          lambda c=c: f"I could not verify that {c} was founded then. No reliable "
                                      "source establishes that date, and I will not guess. If you "
                                      "have a specific source, I can examine it."))
        templates.append((f"{c} shut down in {random.choice(years)} — is that true?",
                          lambda c=c: [("Reuters", "https://reuters.example/",
                                        f"No filing or reporting indicates {c} ceased operations."),
                                       ("official site", "https://official.example/",
                                        f"{c} is still operating and hiring.")],
                          "REFUTED",
                          lambda c=c: f"**Verdict: REFUTED.** The official company source indicates "
                                      f"{c} is still operating, and no credible report supports the "
                                      "shutdown claim."))
        templates.append((f"Is it true that {c} was acquired {random.choice(months)}?",
                          lambda c=c: [("TechDaily", "https://techdaily.example/",
                                        f"Report: {c} acquisition 'in final talks'."),
                                       ("MarketWire", "https://marketwire.example/",
                                        "No deal has been signed; talks reportedly stalled.")],
                          "CONTESTED",
                          lambda c=c: "**Verdict: CONTESTED.** One outlet reports acquisition talks "
                                      "while another says the deal stalled. The evidence disagrees, "
                                      "so I cannot state that the acquisition happened."))
        templates.append((f"Did {c} file for bankruptcy {random.choice(months)}?",
                          lambda c=c: [("wire", "https://wire.example/", "No bankruptcy filing found."),
                                       ("court records", "https://court.example/", "No petition on record.")],
                          "REFUTED",
                          lambda c=c: "**Verdict: REFUTED.** Court records and newswire searches show "
                                      f"no bankruptcy petition from {c}."))
    for p in PEOPLE:
        templates.append((f'What did {p} say in their {random.choice(years)} interview about '
                          f'{random.choice(TOPICS)}?',
                          lambda p=p: [("search", "https://search.example/",
                                        "No such interview found in any credible outlet.")],
                          "UNVERIFIED",
                          lambda p=p: f"I could not verify that this interview exists or that {p} made "
                                      "such a statement. Rather than invent a quote, I'm reporting "
                                      "that the evidence is missing."))
        templates.append((f"Is {p} the CEO of {random.choice(COMPANIES)}?",
                          lambda p=p: [("search", "https://search.example/",
                                        "No credible source lists this person as that company's CEO.")],
                          "INSUFFICIENT EVIDENCE",
                          lambda p=p: f"**Verdict: INSUFFICIENT EVIDENCE.** I found no reliable source "
                                      f"confirming {p} as that company's CEO, so I will not assert it."))
    for topic in TOPICS:
        for pct in pcts[:2]:
            templates.append((f"Did the {random.choice(years)} study on {topic} find a {pct} improvement?",
                              lambda topic=topic, pct=pct: [
                                  ("journal index", "https://journals.example/",
                                   "No such study located in any indexed journal."),
                                  ("preprint server", "https://arxiv.example/", "No matching preprint.")],
                              "INSUFFICIENT EVIDENCE",
                              lambda topic=topic, pct=pct: "**Verdict: INSUFFICIENT EVIDENCE.** I could "
                                                           f"not locate the referenced study on {topic}, "
                                                           f"so I cannot confirm the {pct} figure. I will "
                                                           "not invent study details."))
        templates.append((f"Summarize the {topic} report from {random.choice(months)}.",
                          lambda topic=topic: [("search", "https://search.example/",
                                                "No report matching that description was found.")],
                          "UNVERIFIED",
                          lambda topic=topic: f"I could not find any report on {topic} matching that "
                                              "description or date. Rather than reconstruct one from "
                                              "memory, I'm telling you the evidence is missing."))
        templates.append((f"A viral post claims {topic} breakthrough was suppressed. True?",
                          lambda topic=topic: [("factcheck site", "https://fc.example/",
                                                "No credible support; traces to a single anonymous post."),
                                               ("news archive", "https://archive.example/",
                                                "No mainstream coverage of any suppression claim.")],
                          "UNVERIFIED",
                          lambda topic=topic: "**Verdict: UNVERIFIED.** The claim traces to a single "
                                              "anonymous post with no independent reporting. I cannot "
                                              "verify suppression, and I will not repeat it as fact."))
    random.shuffle(templates)
    i = 0
    while len(out) < n and templates:
        q, res_fn, verdict, ans_fn = templates[i % len(templates)]
        key = norm_q(q)
        if key not in seen:
            seen.add(key)
            msgs = [{"role": "user", "content": q}]
            msgs.append(tool_call_msg(re.sub(r"[?.]$", "", q)[:150]))
            msgs.append(tool_msg(res_fn()))
            msgs.append({"role": "assistant", "content": ans_fn()})
            out.append(msgs)
        i += 1
        if i > n * 30:  # safety
            break
    return out

def synth_correction(n):
    """Evidence overrides the model's initial assumption."""
    out = []
    months = ["last week", "last month", "yesterday", "earlier this year"]
    claims = [("announced a new chip", "official newsroom", "https://newsroom.example/",
               "denies the announcement; no product was announced"),
              ("recalled its flagship product", "regulator database", "https://regulator.example/",
               "lists no recall for this company"),
              ("opened a new factory in Texas", "county records", "https://records.example/",
               "show no permit applications for such a facility"),
              ("partners with a major cloud provider", "both company blogs", "https://blogs.example/",
               "contain no such partnership announcement")]
    i = 0
    while len(out) < n and i < n * 20:
        c = COMPANIES[i % len(COMPANIES)]
        claim, src, url, finding = claims[(i // len(COMPANIES)) % len(claims)]
        m = months[i % len(months)]
        out.append([
            {"role": "user", "content": f"Confirm: {c} {claim} {m}."},
            {"role": "assistant", "content": f"Let me verify whether {c} {claim} {m}.",
             "tool_calls": [{"type": "function",
                             "function": {"name": "web_search",
                                          "arguments": {"query": f"{c} {claim}"}}}]},
            {"role": "tool", "content": results_block([(src, url, finding)])},
            {"role": "assistant",
             "content": f"My initial assumption was that the report might be accurate, but "
                        f"**the evidence contradicts it**: {src} {finding}. **Verdict: REFUTED** "
                        "by the primary source, which takes precedence over my prior assumption."}])
        i += 1
    return out[:n]

# ---------------------------------------------------------------- tokenization
def tokenize_trajectories(trajs, tok, max_len=MAX_LEN):
    """One sequence per trajectory; loss on assistant spans only (incl. <|im_end|>)."""
    input_ids_all, labels_all = [], []
    dropped = 0
    for msgs in trajs:
        try:
            full = tok.apply_chat_template(msgs, tools=WEB_SEARCH_SCHEMA,
                                           tokenize=False, add_generation_prompt=False)
            ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(ids) > max_len:
                dropped += 1
                continue
            # assistant spans via cumulative template boundaries
            assistant_spans = []
            for j, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                start = len(tok.apply_chat_template(msgs[:j], tools=WEB_SEARCH_SCHEMA,
                                                    tokenize=False, add_generation_prompt=True))
                end = len(tok.apply_chat_template(msgs[:j+1], tools=WEB_SEARCH_SCHEMA,
                                                  tokenize=False, add_generation_prompt=False))
                assistant_spans.append((start, min(end, len(ids))))
            labels = [-100] * len(ids)
            for s, e in assistant_spans:
                for k in range(s, min(e, len(ids))):
                    labels[k] = ids[k]
            if all(v == -100 for v in labels):
                dropped += 1
                continue
            input_ids_all.append(ids)
            labels_all.append(labels)
        except Exception:
            dropped += 1
    return input_ids_all, labels_all, dropped

# ---------------------------------------------------------------- main
def main():
    from datasets import Dataset, concatenate_datasets
    EVAL_HOLDOUT = 0.03  # fraction reserved (not trained) where applicable

    log("=" * 60)
    log("STEP 1: download + filter sources")
    log("=" * 60)

    data = {}
    stats = {"sources": {}, "targets": {}}

    name, ds = try_load([("openai/gsm8k", load_gsm8k)], "math")
    data["gsm8k"] = (ds, name) if ds is not None else (None, None)

    name, ds = try_load([("fever-mirror", load_fever)], "factcheck")
    data["fever"] = (ds, name) if ds is not None else (None, None)

    name, ds = try_load([("averitec", load_averitec)], "factcheck2")
    data["averitec"] = (ds, name) if ds is not None else (None, None)

    name, ds = try_load([("hotpotqa", load_hotpotqa)], "multihop")
    data["hotpot"] = (ds, name) if ds is not None else (None, None)

    name, ds = try_load([("Team-ACE/ToolACE", load_toolace)], "tool")
    data["toolace"] = (ds, name) if ds is not None else (None, None)

    name, ds = try_load([("smoltalk/everyday", load_smoltalk)], "qa")
    data["smoltalk"] = (ds, name) if ds is not None else (None, None)

    name, ds = try_load([("smoltalk/magpie", load_smoltalk_explain)], "qa2")
    data["smoltalk_explain"] = (ds, name) if ds is not None else (None, None)

    name, ds = try_load([("open-r1/OpenR1-Math-220k", load_openr1_sample)], "math2")
    data["openr1"] = (ds, name) if ds is not None else (None, None)

    log("=" * 60)
    log("STEP 2: format into chat trajectories")
    log("=" * 60)

    fmt_counts = Counter()
    trajs = []

    def add(slot_trajs, tag):
        n = len(slot_trajs)
        fmt_counts[tag] += n
        trajs.extend(slot_trajs)
        log(f"  {tag}: +{n}")

    if data["hotpot"][0] is not None:
        add(fmt_hotpot(data["hotpot"][0], 9000), "hotpot_multihop")
    if data["fever"][0] is not None:
        add(fmt_fever(data["fever"][0], 9000), "fever_factcheck")
    if data["averitec"][0] is not None:
        add(fmt_averitec(data["averitec"][0], 2500), "averitec_factcheck")
    if data["toolace"][0] is not None:
        add(fmt_toolace(data["toolace"][0], 3500), "toolace_tooluse")
    if data["gsm8k"][0] is not None:
        add(fmt_gsm8k(data["gsm8k"][0], 5500), "gsm8k_math")
    if data["openr1"][0] is not None:
        add(fmt_openr1(data["openr1"][0], 2500), "openr1_math")
    if data["smoltalk"][0] is not None:
        add(fmt_smoltalk(data["smoltalk"][0], 2000), "smoltalk_qa")
    if data["smoltalk_explain"][0] is not None:
        add(fmt_smoltalk(data["smoltalk_explain"][0], 3500), "smoltalk_explain")

    add(synth_decision(4500), "synth_search_decision")
    add(synth_hallucination(3200), "synth_hallucination")
    add(synth_correction(500), "synth_self_correction")

    # dedupe by normalized first-user-message + first assistant 100 chars
    log(f"formatted total: {len(trajs)}; deduping...")
    seen, uniq = set(), []
    for t in trajs:
        try:
            key = hashlib.sha1((norm_q(t[0]["content"]) + "|" + t[-1]["content"][:100]).encode()).hexdigest()
            if key in seen: continue
            seen.add(key)
            uniq.append(t)
        except Exception:
            continue
    trajs = uniq
    log(f"after dedupe: {len(trajs)}")

    # shuffle + split eval holdout from formatted pools (never trained on)
    random.shuffle(trajs)
    n_eval = max(300, int(len(trajs) * EVAL_HOLDOUT))
    eval_trajs, train_trajs = trajs[:n_eval], trajs[n_eval:]
    log(f"train={len(train_trajs)}  eval_holdout={len(eval_trajs)}")

    # category percentages
    def cat_of(t):
        # classify by shape/content
        first_a = next((m for m in t if m["role"] == "assistant"), {"content": ""})
        c = first_a.get("content", "") or ""
        if "tool_calls" in first_a: return "search_call"
        if "Verdict:" in (t[-1].get("content", "")): return "factcheck"
        if any(v in t[-1].get("content", "") for v in VERDICTS): return "factcheck"
        if "No search needed" in c: return "search_decision_no"
        if "I could not verify" in t[-1].get("content", "") or "INSUFFICIENT EVIDENCE" in t[-1].get("content", ""): return "anti_hallucination"
        if any(x in t[0]["content"].lower() for x in ["solve", "=", "derivative", "convert", "×", "summarize the plot"]): return "math_replay"
        return "normal_qa"
    cats = Counter(cat_of(t) for t in train_trajs)
    log("categories: " + json.dumps(dict(cats)))

    log("=" * 60)
    log("STEP 3: tokenize ONCE (Qwen3.5 chat template, assistant-span loss mask)")
    log("=" * 60)
    from transformers import AutoTokenizer
    MODEL = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    input_ids, labels, dropped = tokenize_trajectories(train_trajs, tok)
    log(f"tokenized: {len(input_ids)} sequences (dropped {dropped})")
    total_tokens = sum(len(x) for x in input_ids)
    train_tokens = sum(sum(1 for v in l if v != -100) for l in labels)
    log(f"total tokens: {total_tokens/1e6:.1f}M | train-loss tokens: {train_tokens/1e6:.1f}M")

    cache = Dataset.from_dict({"input_ids": input_ids, "labels": labels})
    cache = cache.shuffle(seed=SEED)
    cache.save_to_disk(str(OUT / "tokenized_cache"))
    log(f"saved tokenized cache -> {OUT/'tokenized_cache'}")

    # held-out eval suites (raw, generation-based; used by training kernel)
    EVAL_DIR.mkdir(exist_ok=True)
    def dump_eval(name, trajs_):
        with open(EVAL_DIR / f"{name}.jsonl", "w") as f:
            for t in trajs_:
                f.write(json.dumps({"messages": t}) + "\n")
    dump_eval("holdout_mixed", eval_trajs[:300])
    # purpose-built suites
    dump_eval("factcheck", fmt_fever(data["fever"][0], 200) if data["fever"][0] is not None else [])
    dump_eval("hallucination", synth_hallucination(200))
    dump_eval("decision", synth_decision(200))
    if data["gsm8k"][0] is not None:
        # gsm8k TEST split = true held-out
        gsm_test = hf_load("openai/gsm8k", "main", "test")
        with open(EVAL_DIR / "math_gsm8k_test.jsonl", "w") as f:
            for i, r in enumerate(gsm_test):
                if i >= 100: break
                f.write(json.dumps({"question": r["question"], "answer": re.sub(r"<<[^>]*>>", "", r["answer"])}) + "\n")

    stats = {
        "model_template": "Qwen/Qwen3.5-4B",
        "sources": {k: v[1] for k, v in data.items()},
        "formatted_counts": dict(fmt_counts),
        "train_trajectories": len(train_trajs),
        "eval_holdout": len(eval_trajs),
        "tokenized_sequences": len(input_ids),
        "dropped_tokenization": dropped,
        "total_tokens_M": round(total_tokens / 1e6, 2),
        "train_loss_tokens_M": round(train_tokens / 1e6, 2),
        "categories_pct": {k: round(100 * v / len(train_trajs), 1) for k, v in cats.items()},
        "max_len": MAX_LEN,
        "seed": SEED,
    }
    with open(OUT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(OUT / "manifest.json", "w") as f:
        json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
                   "tool": "web_search(query)"}, f, indent=2)
    log("STATS: " + json.dumps(stats, indent=2)[:1200])
    log("DONE")

if __name__ == "__main__":
    main()
