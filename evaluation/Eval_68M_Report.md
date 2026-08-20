# Evaluation Report — 68.7M Dense TransformerLM (50k steps, ~10 epochs)

**Model:** 68.7M-param dense decoder-only Transformer — RMSNorm, SwiGLU, RoPE,
6 layers, 8 heads, d_model 576, d_ff 2304, vocab 32k (SentencePiece), 2048 context.
**Checkpoint:** `roronoazoro3008/research-v2-checkpoints` → `resume.pt` (step 50,000).
**Run:** 50,000 steps × 65,536 tokens/step = **3.28B tokens** ≈ **10 epochs** over the
~319M-token research/coding corpus. Trained on Kaggle T4 GPUs across two accounts
(`tomiokasan` + `roronoazoro3008`, split by the weekly 30 h GPU quota) — total
**~34.5 h of GPU time** (final 41k→50k push: 6.31 h).
**Final metrics:** best val loss **4.5870**, val PPL **≈ 98.2**.

Evaluation script: `evaluation/eval_68m_final.py` — loads the checkpoint with the
exact training architecture (`training/kaggle/kaggle_pipeline.py` TransformerLM,
68,721,984 params) and the exact corpus tokenizer.

---

## 1. Sample Generations (temperature 0.8, top-k 50, top-p 0.95, 200 new tokens)

### Prompt 1: "The history of artificial intelligence begins with"
> The history of artificial intelligence begins with discovery an of and
> 03 98 The of. of.1882
> 4. ...612-.12.7.276.768...157-5.

### Prompt 2: "In quantum computing, the fundamental unit is"
> In quantum computing, the fundamental unit is charge a of quantum and a complex with complex,-inplete. generalivity the of is in quantum with complex,inite.
> The,in,,in are and,in,;, are

### Prompt 3: "Python is a programming language that"
> Python is a programming language that not related programming.
> ## The of  is same a programming of that.

### Prompt 4: "def fibonacci(n):"
> def fibonacci(n): return,
> tle[],

### Prompt 5: "The theory of general relativity describes"
> The theory of general relativity describes relat.
> In relat, is, the of which the is of isic in of. if is in,is inisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisisis...

### Prompt 6: "import numpy as np\n\ndef mean(x):"
> import numpy as np
>
> def mean(x):
>  =..(1
> ...
> ...
> ...
>
> eters y
>  x npn
>  nparray nparrayarray
>  np nparrayarrayarray nparray
>  nparray nparrayarrayarrayarrayarrayarrayintint nparrayint
>  nparrayarrayarrayarrayintintarrayarrayarrayarrayintintarrayarrayint
> ...

---

## 2. Findings

| Metric | Value |
|---|---|
| Steps | 50,000 / 50,000 |
| Tokens | 3,276,800,000 (3.28B) |
| Epochs | ~10 (over ~319M-token corpus) |
| Best val loss | 4.5870 |
| Best val PPL | ≈ 98.2 |
| Training time | ~34.5 h GPU (two accounts, quota-split) |

**What the model learned:**
- Word-level statistics of the corpus (common tokens like *of, the, is, in, a* appear fluently).
- Formatting/structural conventions: code prompts (*def*, *import numpy as np*) produce
  indentation, blank lines and function-body punctuation, but not valid code.
- **No** long-range coherence, factual recall, or grammar beyond a few tokens.

**Observed failure mode (consistent with all 6 prompts):**
1. 3–4 tokens of plausible continuation matching the prompt's surface style
   ("begins with discovery", "the fundamental unit is charge", "not related programming").
2. Then collapse into hallucination: random digit/date fragments, punctuation runs,
   and — in the worst cases — hard repetition of a single token or syllable
   (`isisisisisis...`, `nparray nparrayarrayarray...`) until the 200-token budget runs out.

---

## 3. Conclusion — 68M parameters is not enough

The training budget was respectable (50k steps, 3.28B tokens ≈ 10 epochs, ~34.5 h GPU),
and the run was stable — val loss fell to 4.5870 with no divergence. Despite this:

- **68M params is simply too small** for a general-purpose LM to compress the
  underlying patterns of language/code into usable representations. The capacity is
  absorbed by token-level statistics; there is no room left for syntax, semantics,
  or long-range structure.
- The **hallucination-vs-correct-token ratio** is the sign: the model outputs correct
  tokens for 3–4 tokens and then hallucinates "*like crazy*" — it has learned *plausible
  token transitions*, not *meaning*.
- Perplexity ≈ 98 confirms this: a near-random-ish token distribution that beats
  uniform guessing but fails any real linguistic test.

**Implication:** scaling the *data* further won't fix this — the bottleneck is model
capacity. This is exactly why the project pivoted to:
- **Distillation** (already in progress) — using a much larger teacher to *generate
  the data* for our model, so the small model learns from compressed, high-quality
  signal instead of raw web text.
- **Architecture experiments** that were tested on this run family: **Muon** optimizer,
  **MoE** (sparse experts), **sparse/block attention** — all under the same 70M-class
  budget.

**Next goal:** instead of a generalist 68M model trying to cover *many domains* (and
failing at all of them), train a model that is **specialized / expert in one domain** —
with distillation-generated data for that domain, a much lower effective vocabulary
and far fewer concepts to compress, giving the same 68-70M capacity a real chance
of learning actual patterns rather than token-frequency artifacts.

---

*Generated 2026-08-14 · device: Apple MPS · eval script: `evaluation/eval_68m_final.py`*