"""Generate report diagrams for the on-device Qwen Android mini-project."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = "/Users/rohit/Documents/Sem5_project/MiniProject_Report_QwenAndroid/images"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})


def box(ax, xy, w, h, text, fc="#DCE9FF", ec="#1A73E8", fs=9):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True)


def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.2, color="#333333"))


def finalize(ax, path, w=10, h=6):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig = ax.figure
    fig.set_size_inches(w, h)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------- 1. System architecture ----------------
fig, ax = plt.subplots()
H = 8
box(ax, (3.6, 6.8), 2.8, 0.8, "User (Android UI)")
arrow(ax, (5.0, 6.8), (5.0, 6.1))
box(ax, (3.6, 5.3), 2.8, 0.8, "ChatActivity\n(messages, thinking cards)")
arrow(ax, (5.0, 5.3), (5.0, 4.6))
box(ax, (3.6, 3.8), 2.8, 0.8, "Tool loop\n(parse think/tool_call)")
arrow(ax, (3.6, 4.2), (2.6, 4.2))
box(ax, (0.2, 3.8), 2.2, 0.8, "WebSearch\n(DDG + Wikipedia)", fc="#E8F0FE")
arrow(ax, (2.6, 4.6), (3.6, 4.6))
arrow(ax, (5.0, 3.8), (5.0, 3.1))
box(ax, (3.6, 2.3), 2.8, 0.8, "llama.cpp JNI\n(InferenceEngine)", fc="#FFF3D6", ec="#B7791F")
arrow(ax, (5.0, 2.3), (5.0, 1.6))
box(ax, (3.6, 0.8), 2.8, 0.8, "Qwen3.5 ORPO-v4\nQ4_K_M GGUF (2.5 GB)", fc="#E6F4EA", ec="#1E8E3E")
box(ax, (7.0, 3.8), 2.6, 0.8, "Model pipeline\n(merge + convert)", fc="#F3E8FD", ec="#7B1FA2")
arrow(ax, (7.0, 4.2), (6.4, 4.2))
finalize(ax, f"{OUT}/system_architecture.png", h=H)

# ---------------- 2. ER diagram ----------------
fig, ax = plt.subplots()
H = 8
ents = [
    ((0.4, 6.6), "User\nuser_id, name"),
    ((3.4, 6.6), "ChatSession\nsession_id, user_id,\nstarted_at"),
    ((6.4, 6.6), "Message\nmessage_id, session_id,\nrole, text, time"),
    ((0.4, 4.4), "ModelArtifact\nmodel_id, name,\nquant, size_gb"),
    ((3.4, 4.4), "ToolCall\ntoolcall_id, message_id,\ntool, query"),
    ((6.4, 4.4), "SearchResult\nresult_id, toolcall_id,\nsource, snippet"),
    ((3.4, 2.2), "ThinkingTrace\ntrace_id, message_id,\nreasoning_text"),
]
for (x, y), t in ents:
    box(ax, (x, y), 2.4, 1.0, t, fc="#E8F0FE", fs=8)
rels = [((1.6, 6.6), (3.4, 7.1)), ((4.6, 6.6), (6.4, 7.1)),
        ((1.6, 4.4), (3.4, 4.9)), ((4.6, 4.4), (6.4, 4.9)),
        ((4.6, 5.4), (4.6, 3.2))]
for a, b in rels:
    arrow(ax, a, b)
ax.text(2.5, 7.35, "1:N", fontsize=8)
ax.text(5.5, 7.35, "1:N", fontsize=8)
ax.text(2.5, 5.15, "1:N", fontsize=8)
ax.text(5.5, 5.15, "1:N", fontsize=8)
finalize(ax, f"{OUT}/er_diagram.png", h=H)

# ---------------- 3. UML activity (vertical flow) ----------------
fig, ax = plt.subplots()
steps = [
    "App launch:\nauto-load GGUF",
    "User sends\nmessage",
    "Generate:\ncollect tokens",
    "Parse think +\ntool_call?",
    "Tool call?\nYES: web_search\nNO: show answer",
    "Send tool result;\ngenerate final",
    "Render thinking card\n+ answer",
]
y = 7.4
for i, s in enumerate(steps):
    box(ax, (3.5, y), 3.0, 0.75, s, fc="#E8F0FE" if i not in (4,) else "#FFF3D6",
        ec="#1A73E8" if i not in (4,) else "#B7791F", fs=8)
    if i < len(steps) - 1:
        arrow(ax, (5.0, y), (5.0, y - 0.35))
    y -= 1.1
finalize(ax, f"{OUT}/activity_uml.png", h=8.6)

# ---------------- 4. DFD level 1 ----------------
fig, ax = plt.subplots()
H = 8
box(ax, (0.3, 6.8), 1.8, 0.7, "User", fc="#F1F3F4", ec="#5F6368")
box(ax, (7.9, 6.8), 1.8, 0.7, "Web APIs", fc="#F1F3F4", ec="#5F6368")
procs = [
    ((3.9, 6.8), "P1\nChat UI"),
    ((3.9, 5.5), "P2\nThink/Tool parse"),
    ((3.9, 4.2), "P3\nWeb search"),
    ((3.9, 2.9), "P4\nllama.cpp inference"),
]
for (x, y), t in procs:
    b = FancyBboxPatch((x, y), 2.2, 0.7,
                       boxstyle="round,pad=0.02,rounding_size=0.3",
                       fc="#DCE9FF", ec="#1A73E8", lw=1.2)
    ax.add_patch(b)
    ax.text(x + 1.1, y + 0.35, t, ha="center", va="center", fontsize=8)
box(ax, (7.9, 4.2), 1.8, 0.7, "D1: GGUF\nmodel file", fc="#E6F4EA", ec="#1E8E3E")
box(ax, (0.3, 4.2), 1.8, 0.7, "D2: Chat\nhistory", fc="#E6F4EA", ec="#1E8E3E")
flows = [((2.1, 7.15), (3.9, 7.15)), ((5.0, 6.8), (5.0, 6.2)),
         ((5.0, 5.5), (5.0, 4.9)), ((6.1, 4.55), (7.9, 4.55)),
         ((7.9, 6.0), (6.1, 4.0)), ((5.0, 2.9), (5.0, 2.3)),
         ((3.9, 3.25), (2.1, 4.55)), ((2.1, 5.9), (3.9, 5.9))]
for a, b in flows:
    arrow(ax, a, b)
finalize(ax, f"{OUT}/dfd_level1.png", h=H)

# ---------------- 5. Gantt ----------------
tasks = [
    ("Post-training (QLoRA→ORPO)", 0, 3),
    ("Merge + GGUF conversion", 3, 4),
    ("Android Studio + SDK setup", 4, 5),
    ("App scaffold (llama.cpp)", 5, 6),
    ("Chat UI + think/tool loop", 6, 8),
    ("Dark mode + polish", 8, 9),
    ("On-device testing", 7, 10),
    ("Report writing", 9, 11),
]
fig, ax = plt.subplots(figsize=(10, 4.5))
yt = np.arange(len(tasks))
for i, (name, s, e) in enumerate(tasks):
    ax.barh(i, e - s, left=s, height=0.5, color="#1A73E8")
    ax.text(s, i, f"  {name}", va="center", fontsize=9)
ax.set_yticks([])
ax.set_xlabel("Project week")
ax.set_xlim(0, 11)
ax.set_title("Project Schedule (Gantt Chart)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/gantt_chart.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- 6. Benchmark charts (README screening numbers) ----------------
tasks7 = ["MMLU-Pro CS", "TruthfulQA", "GSM8K", "HellaSwag", "BBH-LD5", "IFEval", "ARC-C"]
base = [0.0, 57.38, 0.0, 70.0, 30.0, 22.22, 30.0]
orpo = [30.0, 43.86, 60.0, 50.0, 20.0, 22.22, 40.0]
x = np.arange(len(tasks7))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, base, 0.4, label="Base Qwen3.5-4B", color="#9AA0A6")
ax.bar(x + 0.2, orpo, 0.4, label="ORPO-v4 final", color="#1A73E8")
ax.set_xticks(x)
ax.set_xticklabels(tasks7, rotation=15, ha="right")
ax.set_ylabel("Score (%)")
ax.set_title("7-task screening (n=10 each): Base vs ORPO-v4")
ax.legend()
ax.set_ylim(0, 80)
fig.tight_layout()
fig.savefig(f"{OUT}/average_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)

cats = ["GSM8K flex", "GSM8K strict", "IFEval-P strict", "IFEval-I strict",
        "IFEval-P loose", "IFEval-I loose"]
base50 = [6.0, 0.0, 14.0, 30.67, 14.0, 30.67]
orpo50 = [66.0, 60.0, 14.0, 30.67, 18.0, 33.33]
x = np.arange(len(cats))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, base50, 0.4, label="Base Qwen3.5-4B", color="#9AA0A6")
ax.bar(x + 0.2, orpo50, 0.4, label="ORPO-v4 final", color="#1E8E3E")
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=15, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Paired 50-example check: GSM8K + IFEval (Base vs ORPO-v4)")
ax.legend()
ax.set_ylim(0, 75)
fig.tight_layout()
fig.savefig(f"{OUT}/per_query_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("done:", sorted(os.listdir(OUT)))
